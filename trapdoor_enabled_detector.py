import sklearn.preprocessing
from utils import *


class TrapdoorEnabledDetector:
    def __init__(self, dataset, model_latent, target, target_layer, triggers, gen_train, gen_test, exp_dir, fp_percentile=5,
                 estimated_signatures=[], distinct_method=None, debug=False,
                 sampling_ratio=1.0, n_repeat=1, pool_size=1, save_score=False):
        assert distinct_method in ['orthogonal', 'target5', 'other5']
        print("----------------start building detector for label %d----------------" % target)
        self.model_latent = model_latent
        self.protected_label = target
        self.fp_percentile = fp_percentile
        self.distinct_method = distinct_method
        self.debug = debug
        self.target_layer = target_layer
        self.triggers = triggers
        self.target = target
        self.gen_train = gen_train
        self.gen_test = gen_test
        self.pool_size = pool_size
        self.sampling_ratio = sampling_ratio
        self.n_repeat = n_repeat

        self.detection_score_path = "exp_result/%s/score/score_%s.csv" % (dataset, "%s")
        self.activation_path = "%s/activation_%s_%s.npy" % (exp_dir, target_layer, "%s")

        # get latent representation of begin samples for threshold calculation
        self.latent_clean, self.label_clean, self.l2_norm_clean, acc, target_acc = extract_activation_with_gen(
            self.model_latent, gen_train, self.activation_path % "clean_train", target=target)
        print("clean accuracy: %f, target category acc: %f" % (acc, target_acc))
        self.is_target = (self.label_clean == self.protected_label)
        if debug:
            # calculate intra and inter class similarity
            def get_centroid(latent):
                return normalize(np.median(latent, axis=0))
            latent_clean_target = self.latent_clean[self.is_target][:64]
            print("intra class similarity of target:\n%s" %
                  cosine_similarity(get_centroid(latent_clean_target), latent_clean_target))
            labels = np.unique(self.label_clean)
            label_clean_other = random.choice(labels[labels != self.protected_label])
            latent_clean_other = self.latent_clean[self.label_clean == label_clean_other][:64]
            print("intra class similarity of other:\n%s" %
                  cosine_similarity(get_centroid(latent_clean_other), latent_clean_other))
            len_cut = min(len(latent_clean_target), len(latent_clean_other))
            print("inter class similarity between target and other:\n%s" %
                  cosine_similarity(latent_clean_target[:len_cut], latent_clean_other[:len_cut]))

        self.n_trapdoor = len(triggers)
        if self.n_trapdoor:
            shape = (self.n_trapdoor,) + self.latent_clean.shape[1:]
            self.signatures_init = np.zeros(shape)
            self.signatures_origin = np.zeros(shape)  # before decomposed or selection
        else:
            self.signatures_init = np.array(estimated_signatures)
            self.n_trapdoor = len(estimated_signatures)
        self.signatures = None  # used for current detection
        self.masks = np.ones_like(self.signatures_init, dtype=np.bool)
        self.thresholds = None
        # used for multiple rounds detection
        self.signatures_list = []
        self.masks_list = []
        self.thresholds_list = []

        # random refresh pool
        self.masks_pool = []
        self.signatures_pool = []

        self.trapdoor_acc_record = []
        self.is_effective = np.ones(len(self.signatures_init), dtype=np.bool)  # control which trapdoors involve detetcion
        self.fprs_benign_target = None
        self.fprs_benign = None
        self.united_fpr_benign_target = 0.
        self.united_fpr_benign = 1.
        self.detected_clean = None

        self.extract_signature_and_fill_random_pool()
        if self.sampling_ratio == 1.0:
            self.update_thresholds(verbose=self.debug, save_score=save_score)

    def extract_signature_and_fill_random_pool(self):
        print("==========filling random pool==========")
        self.masks_pool.clear()
        self.signatures_pool.clear()
        if self.sampling_ratio < 1.0:
            pool_candidate_size = (min(self.pool_size * 10, 100), self.latent_clean.shape[-1])
            self.construct_graph_for_subsets_decomposition()
        # extract signatures for each trapdoor
        latent_clean_target = self.latent_clean[self.is_target]
        labels = np.unique(self.label_clean)
        label_clean_other = random.choice(labels[labels != self.protected_label])
        latent_clean_other = self.latent_clean[self.label_clean == label_clean_other][:64]
        if type(self.gen_train) is DirectoryIterator:  # remove all target samples for trapdoor sample generator
            # keep only target filepaths
            keeper = [self.gen_train.classes, self.gen_train.filenames, self.gen_train._filepaths]
            split_directory_iterator_with_condition(self.gen_train, (self.gen_train.classes != self.target))

        for i, trigger in enumerate(self.triggers):
            mask, pattern = trigger["mask"], trigger["pattern"]
            trapdoor_gen, steps = get_trapdoor_gen(self.gen_train, self.target, mask, pattern)
            activation_trapdoor, label, l2_norm, acc, _ = extract_activation_with_gen(
                self.model_latent, trapdoor_gen, self.activation_path % ("trapdoor_%d_%d" % (self.target, i)),
                steps=steps, n=trapdoor_gen.n if type(trapdoor_gen) is NumpyArrayIterator else self.gen_train.n,
                target=self.target)
            print("accuracy of label %d, trapdoor %d: %f" % (self.target, i, acc))
            self.trapdoor_acc_record.append(acc)
            signature = ((activation_trapdoor * l2_norm[:, np.newaxis])[label == self.protected_label]).mean(axis=0)
            signature = normalize(signature)
            if self.debug:
                # calculate target clean and trapdoor similarity
                print("intra class similarity of trapdoor:\n%s" %
                      cosine_similarity(signature, activation_trapdoor[:64]))
                len_cut = min(len(latent_clean_target[:64]), len(activation_trapdoor))
                print("inter class similarity between trapdoor and target:\n%s" %
                      cosine_similarity(latent_clean_target[:64][:len_cut], activation_trapdoor[:len_cut]))
                len_cut = min(len(latent_clean_other), len(activation_trapdoor))
                print("inter class similarity between trapdoor and other:\n%s" %
                      cosine_similarity(latent_clean_other[:len_cut], activation_trapdoor[:len_cut]))
            if self.sampling_ratio < 1.0:
                self.get_decomposed_sigs_and_score(pool_candidate_size, signature, latent_clean_target, activation_trapdoor)

            self.signatures_origin[i] = signature
            if self.distinct_method != 'target5':
                self.signatures_init[i], _ = self.get_distinctive_subset(
                    signature, np.ones_like(signature, dtype=np.bool), verbose=self.debug)
            else:
                self.signatures_init[i] = signature

        if self.sampling_ratio < 1.0:
            self.masks_pool = [msks for msks in np.stack(self.masks_pool).transpose(1, 0, 2)]
            self.signatures_pool = [sigs for sigs in np.stack(self.signatures_pool).transpose(1, 0, 2)]

        if type(self.gen_train) is DirectoryIterator:
            # recover data generator
            set_directory_iterator(self.gen_train, keeper)

    def get_distinctive_subset(self, signature, sampling_mask, verbose=False):
        # get trapdoor distinctive components to avoid feature overlapping which brings false positive rate
        latent_benign_target = self.latent_clean[self.is_target]
        cosine_mean_before = cosine_similarity(sklearn.preprocessing.normalize((signature * sampling_mask).reshape(1, -1)),
                                               sklearn.preprocessing.normalize(latent_benign_target * sampling_mask)).mean()
        if self.distinct_method == 'orthogonal':
            # orthogonal decomposition way
            latent_benign_target_mean = (self.latent_clean * self.l2_norm_clean.reshape((-1, 1)))[self.is_target].mean(axis=0)
            signature[sampling_mask], _ = orthogonal_decomposition(signature[sampling_mask],
                                                                   latent_benign_target_mean[sampling_mask])
        signature[sampling_mask] = sklearn.preprocessing.normalize(signature[sampling_mask].reshape(1, -1))[0]
        cosine_mean_after = cosine_similarity(signature[sampling_mask],
                                              normalize(latent_benign_target[:, sampling_mask])).mean()
        if verbose:
            print("%d neurons are kept, average cosine similarity on benign target before/after is %f/%f" % (
                sampling_mask.sum(), cosine_mean_before, cosine_mean_after
            ))
        return signature, sampling_mask

    def get_thresholds(self, scores_clean):
        fp_of_benign_target = True
        if self.distinct_method == 'other5':
            fp_of_benign_target = False

        if fp_of_benign_target:
            return np.percentile(scores_clean[self.is_target], 100 - self.fp_percentile, axis=0)
        else:
            return np.percentile(scores_clean[~self.is_target], 100 - self.fp_percentile, axis=0)
            # np.percentile(scores_clean, 100 - self.fp_percentile, axis=0)

    def update_thresholds(self, save_score=False, verbose=False):
        if self.signatures is None:  # +multiple_random
            if self.sampling_ratio < 1.0:
                if len(self.signatures_pool) == 0:
                    self.extract_signature_and_fill_random_pool()
                self.masks = self.masks_pool.pop()
                self.signatures = self.signatures_pool.pop()
            else:
                self.signatures = self.signatures_init
        scores_clean = self.get_scores(self.latent_clean)
        self.thresholds = self.get_thresholds(scores_clean)
        detected, detected_united = self.eval_scores(scores_clean)
        self.fprs_benign = detected.mean(axis=0)
        self.united_fpr_benign = detected_united.mean()
        self.fprs_benign_target = detected[self.is_target].mean(axis=0)
        self.united_fpr_benign_target = detected_united[self.is_target].mean()
        if verbose:
            print(">> thresholds are %s, fpr benign (target) is %f (%f)" %
                  (self.thresholds, self.united_fpr_benign, self.united_fpr_benign_target))
        if save_score:
            np.savetxt(self.detection_score_path % ('benign_overall_%s_%d' % (
                self.distinct_method, self.protected_label)), scores_clean, delimiter=",")
            np.savetxt(self.detection_score_path % ('benign_target_%s_%d' % (
                self.distinct_method, self.protected_label)), scores_clean[self.is_target], delimiter=",")
        return detected_united

    def detect_latent(self, latent_eval, refresh=False):
        if self.signatures is None or refresh and self.sampling_ratio < 1.0:
            self.refresh_random_state()
        if self.sampling_ratio < 1.0:
            scores, detected, detected_united = [], [], []
            for sig, msk, thres in zip(self.signatures_list, self.masks_list, self.thresholds_list):
                self.signatures = sig
                self.masks = msk
                self.thresholds = thres
                scores_round, detected_round, detected_united_round = self.detect_latent_round(latent_eval)
                scores.append(scores_round)
                detected.append(detected_round)
                detected_united.append(detected_united_round)

            scores, detected, detected_united = np.stack(scores, axis=-1), np.stack(detected, axis=-1),\
                   np.stack(detected_united, axis=-1).sum(axis=-1) > 0
        else:
            scores, detected, detected_united = self.detect_latent_round(latent_eval)

        print("numbers of detected samples for each signatures are %s" % detected.sum(axis=0).squeeze())
        print("mean scores of detected samples for each signatures are %s" %
              [scores[:, i][detected[:, i]].mean() if detected[:, i].sum() > 0 else "N/A" for i in range(self.is_effective.sum())])
        return scores, detected, detected_united

    def update_random_state(self, sampling_ratio, n_repeat, pool_size):  # switch between different ratio
        self.sampling_ratio = sampling_ratio
        self.n_repeat = n_repeat
        self.pool_size = pool_size
        self.fp_percentile = 10. / (self.n_trapdoor * self.n_repeat)
        if sampling_ratio < 1.0:
            self.extract_signature_and_fill_random_pool()
        else:
            self.signatures = self.signatures_init
            self.masks = np.ones_like(self.signatures_init)

    def upsample_detect(self, upsample_size):
        for _ in range(int(self.n_repeat*(1+upsample_size))):
            self.update_thresholds(verbose=self.debug)
            self.signatures_list.append(self.signatures.copy())  # copy paste
            self.masks_list.append(self.masks.copy())
            self.thresholds_list.append(self.thresholds)

    def refresh_random_state(self):
        self.signatures_list.clear()
        self.masks_list.clear()
        self.thresholds_list.clear()
        detected_united_clean = []
        for _ in range(self.n_repeat):
            # only need do when random_sampling
            detected_united_clean_round = self.update_thresholds(verbose=self.debug)
            detected_united_clean.append(detected_united_clean_round)
            self.signatures_list.append(self.signatures.copy())  # copy paste
            self.masks_list.append(self.masks.copy())
            self.thresholds_list.append(self.thresholds)
        print("-----------united detector info-----------")
        detected_united_clean = np.stack(detected_united_clean, axis=-1)
        print("fprs benign target per round are %s" % detected_united_clean[self.is_target].mean(axis=0))
        detected_united_clean = (detected_united_clean.sum(axis=-1) > 0)
        self.united_fpr_benign = detected_united_clean.mean(axis=0)
        self.united_fpr_benign_target = detected_united_clean[self.is_target].mean(axis=0)
        print("fpr united(target) is %f(%f)" % (self.united_fpr_benign, self.united_fpr_benign_target))
        self.detected_clean = detected_united_clean

    def detect_latent_round(self, latent_eval):
        if len(self.signatures):
            scores = self.get_scores(latent_eval)
            detected, detected_united = self.eval_scores(scores)
            return scores, detected, detected_united
        else:
            return np.array([]), np.array([[]]), np.array([])

    def detect_input(self, input_eval, refresh=False):
        latent_eval, predict_eval, _ = self.model_latent.predict_on_batch(input_eval)
        return self.detect_latent(latent_eval, refresh=refresh), (latent_eval, predict_eval)

    def get_scores(self, latent_eval):
        # assume that latent_eval have been normalized before
        sig_effective = self.signatures
        latent_eval_expanded = latent_eval[:, np.newaxis, :]
        if self.masks[0].sum() == len(self.masks[0]):
            return cosine_similarity(sig_effective, latent_eval_expanded)
        else:
            return cosine_similarity(
                sig_effective * self.masks,
                normalize(latent_eval_expanded * self.masks))

    def eval_scores(self, scores_eval):
        detected = (scores_eval > self.thresholds)[:, self.is_effective]
        detected_united = (detected.sum(axis=1) > 0)
        return detected, detected_united


    def construct_graph_for_subsets_decomposition(self, pool_candidate_size):
        # construct the computation graph for random decomposed candidates
        masks_input = K.placeholder(pool_candidate_size)
        signature_input = K.placeholder(pool_candidate_size)
        signature_masked = tf.reshape(tf.boolean_mask(signature_input, masks_input), (pool_candidate_size[0], -1))
        latent_benign = K.placeholder((None, pool_candidate_size[1]))
        latent_trapdoor = K.placeholder((None, pool_candidate_size[1]))
        l2_norm_benign = K.placeholder((None, 1))
        latent_benign_centroid = K.mean(latent_benign * l2_norm_benign, axis=0)
        benign_expanded = tf.repeat(latent_benign_centroid[tf.newaxis, :], pool_candidate_size[0], axis=0)
        benign_masked = tf.reshape(tf.boolean_mask(benign_expanded, masks_input), (pool_candidate_size[0], -1))
        if self.distinct_method == 'orthogonal':
            # orthogonal decomposition
            signature_decomposed = signature_masked - (K.sum(signature_masked * benign_masked, axis=-1) / (
                    K.sum(benign_masked ** 2, axis=-1) + 1e-5))[:, tf.newaxis] * benign_masked
            signature_masked = K.l2_normalize(signature_masked, axis=-1)
            signature_decomposed = K.l2_normalize(signature_decomposed, axis=-1)
        else:
            signature_decomposed = K.l2_normalize(signature_masked, axis=-1)

        def expand_mask_normalize(latent):
            latent_expanded = tf.repeat(latent[:, tf.newaxis, :], pool_candidate_size[0], axis=1)

            latent_masked = tf.reshape(tf.boolean_mask(latent_expanded, masks_input, axis=1),
                                       (tf.shape(latent)[0], pool_candidate_size[0], -1))
            latent_normalized = K.l2_normalize(latent_masked, axis=-1)
            return latent_normalized

        latent_benign_processed = expand_mask_normalize(latent_benign)
        latent_trapdoor_processed = expand_mask_normalize(latent_trapdoor)

        self.f_get_decomposed_sigs_and_score = K.function(
            [masks_input, signature_input, latent_benign, l2_norm_benign, latent_trapdoor],
            [signature_decomposed,
             [K.sum(latent_benign_processed * signature_masked, axis=-1),
              K.sum(latent_benign_processed * signature_decomposed, axis=-1),
              K.sum(latent_trapdoor_processed * signature_masked, axis=-1),
              K.sum(latent_trapdoor_processed * signature_decomposed, axis=-1)]])

    def get_decomposed_sigs_and_score(self, pool_candidate_size, signature, latent_clean_target, activation_trapdoor,
                                      percentile_distinguishable=25):
        # iterate to get enough random sampling subsets
        masks_distinguishable = []
        signatures_distinguishable = []
        n_to_generate = self.pool_size
        masks_candidates = np.ones(pool_candidate_size, np.bool)  # up sample
        while n_to_generate > 0:
            refresh_random_sampling_masks(masks_candidates, self.sampling_ratio)
            signatures_candidates = signature[np.newaxis, :].repeat(pool_candidate_size[0], axis=0)
            sig_masked_decomposed, scores = self.f_get_decomposed_sigs_and_score([
                masks_candidates, signatures_candidates, latent_clean_target[:10000],
                self.l2_norm_clean[self.is_target].reshape((-1, 1))[:10000], activation_trapdoor[:10000]])  #
            index_keep = np.percentile(scores[3], percentile_distinguishable, axis=0) > np.percentile(
                scores[1], 100 - percentile_distinguishable, axis=0)
            if self.debug:
                info = np.stack([s[:, index_keep].mean(axis=0) for s in scores]).transpose()
                if index_keep.sum() > 0:
                    print("cosine similarity to benign & trapdoor before/after:\n", info)
            signatures_candidates[masks_candidates] = sig_masked_decomposed.flatten()
            masks_distinguishable.append(masks_candidates[index_keep][:self.pool_size].copy())
            signatures_distinguishable.append(signatures_candidates[index_keep][:self.pool_size])
            n_to_generate -= min(index_keep.sum(), self.pool_size)
        self.masks_pool.append(np.concatenate(masks_distinguishable, axis=0)[:self.pool_size])
        self.signatures_pool.append(np.concatenate(signatures_distinguishable, axis=0)[:self.pool_size])
