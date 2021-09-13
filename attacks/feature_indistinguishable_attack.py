import numpy as np

from utils import *
import keras
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import euclidean_distances


def optimize_linear(grad, eps, ord=np.inf):
    # update adversarial perturbation in L-inf norm with gradient sign
    if ord == np.inf:
        # Take sign of gradient
        optimal_perturbation = K.sign(grad)
        optimal_perturbation = K.stop_gradient(optimal_perturbation)
    else:
        raise NotImplementedError("currently implement only L-inf norm")
    scaled_perturbation = optimal_perturbation * eps
    return scaled_perturbation


def decomposition(v1, v2):
    dot_product = tf.reduce_sum(v1 * v2, axis=(1, 2, 3))
    return v1 - v2 * tf.reshape(tf.cast(K.less(dot_product, 0), dtype=tf.float32) * dot_product /
                                (tf.reduce_sum(v2 ** 2, axis=(1, 2, 3)) + K.epsilon()), (-1, 1, 1, 1))


def decomposition_np(v1, v2):
    return normalize(v1 - ((v1*v2).sum(axis=-1) / (v2 ** 2).sum(axis=-1))[:, np.newaxis] * v2[np.newaxis, :])


def eval_direction(directions, detected, undetected):
    scores_adv = cosine_similarity(directions, detected[:, np.newaxis, :])
    scores_benign = cosine_similarity(directions, undetected[:, np.newaxis, :])
    score = scores_adv.mean(axis=0) - scores_benign.mean(axis=0)  # using the mean score as distribution distinguishability
    print("scores of candidate directions: ", score)
    return score


def is_cosine_separable(direction, left_distribution, right_distribution):
    scores_left = cosine_similarity(left_distribution, direction)
    scores_right = cosine_similarity(right_distribution, direction)
    return is_separable(scores_right, scores_left)


def is_separable(scores_right, scores_left, percentile=10):
    # percentile control the relaxation of separable, the larger is relaxer
    return np.percentile(scores_right, percentile) - np.percentile(scores_left, 100 - percentile)


class FeatureSpaceAttackStep:
    def __init__(self, fd, debug=False, **kwargs):
        self.fd = fd
        self.attack = None
        self.debug = debug
        self.generate(**kwargs)

    def generate(self, eps_iter, clip_min, clip_max, y_target, ce, **kwargs):
        input_shape = self.fd[0].model_latent.input_shape
        coeff = K.placeholder((len(self.fd), input_shape[0]))
        x = K.placeholder(input_shape)
        self.target_direction, self.cosine_radius, self.target_latent, self.mse_radius = [], [], [], []
        self.evade_direction, self.evade_threshold = [], []
        loss_list, grad_adv_list, outputs_list = [], [], []
        for i, fd in enumerate(self.fd):
            self.target_direction.append(K.variable(fd.target_direction))
            self.cosine_radius.append(K.variable(fd.cosine_radius))
            self.target_latent.append(K.variable(fd.target_latent))
            self.mse_radius.append(K.variable(fd.mse_radius))
            latent_output_tensor, predict_tensor, l2_norm_tensor = fd.model_latent(x)
            loss_cosine = K.sum(latent_output_tensor * self.target_direction[-1], axis=-1)
            loss_mse = keras.losses.mean_squared_error(latent_output_tensor * l2_norm_tensor[:, tf.newaxis],
                                                       self.target_latent[-1])
            if fd.sampling_masks is None:
                loss_cosine_total = loss_cosine
            else:
                latent_output_tensor_sampled = K.l2_normalize(
                    latent_output_tensor[:, tf.newaxis, :] * fd.sampling_masks, axis=-1)
                loss_cosine_sub = K.sum(latent_output_tensor_sampled * fd.target_subdirections, axis=-1)
                loss_cosine_total = loss_cosine + K.sum(K.minimum(loss_cosine_sub, fd.cosine_radius_sub - 0.05), axis=-1)
            reach_target = K.equal(K.argmax(predict_tensor, axis=-1), y_target)
            within_cos_radius = K.greater(loss_cosine, self.cosine_radius[-1])
            within_mse_radius = K.less(loss_mse, self.mse_radius[-1])
            if fd.evade:
                self.evade_direction.append(K.variable(fd.evade_direction))
                self.evade_threshold.append(K.variable(fd.evade_threshold))
                loss_evade = K.sum(latent_output_tensor * self.evade_direction[-1], axis=-1)
                below_threshold = K.less(loss_evade, self.evade_threshold[-1])
                loss = -loss_cosine_total + coeff[i] * loss_evade \
                       + (1 - tf.cast(tf.logical_and(reach_target, within_mse_radius), dtype=tf.float32)) * loss_mse
                if ce:
                    grad_evade, = K.gradients(loss_evade, x)
                    loss_adv = keras.losses.categorical_crossentropy(tf.repeat(
                        to_categorical(y_target, predict_tensor.shape[1])[np.newaxis, :], tf.shape(x)[0], axis=0),
                        predict_tensor)
                    top_2, _ = tf.math.top_k(predict_tensor, 2)
                    max_gap = top_2[:, 0] - top_2[:, 1]
                    grad_adv, = K.gradients((1 - tf.cast(
                        tf.logical_and(reach_target,
                                       K.greater(max_gap, 1 / predict_tensor.shape[1].value)),
                        dtype=tf.float32)) * loss_adv, x)
                    grad_adv_list.append(decomposition(grad_adv, grad_evade))
            else:
                loss = -loss_cosine_total + (
                        1 - tf.cast(tf.logical_and(reach_target, within_mse_radius), dtype=tf.float32)) * loss_mse
            loss_list.append(loss)

            outputs_fd = [predict_tensor, loss_mse, loss_cosine, reach_target, within_cos_radius, within_mse_radius]
            if fd.evade:
                outputs_fd.append([loss_evade, below_threshold])
            if fd.sampling_masks is not None:
                outputs_fd.append(loss_cosine_sub)
            if self.debug:
                # for debug: cosine similarity to oracle trapdoor signature
                latent_output_detect = fd.detector.model_latent(x)[0] if fd.layer_mismatch else latent_output_tensor
                loss_cosine_oracle = K.sum(
                    latent_output_detect[:, tf.newaxis, :] * fd.detector.signatures[fd.detector.is_effective], axis=-1)
                outputs_fd.append(loss_cosine_oracle)
            outputs_list.append(outputs_fd)

        grad, = K.gradients(K.sum(loss_list, axis=0), x)
        if ce:
            for g in grad_adv_list:
                grad += g
        optimal_perturbation = optimize_linear(grad, eps_iter, ord=np.inf)
        adv_x = x - optimal_perturbation
        adv_x = K.clip(adv_x, clip_min, clip_max)  # bound input in valid input space
        outputs = [adv_x, outputs_list]
        self.attack = K.function([x, coeff], outputs)

    def generate_np(self, x_eval, coeff):
        return self.attack([x_eval, coeff])


class FeatureDistribution:
    # modeling the l2 and cosine convex region for target layer
    def __init__(self, detector, model, target_layer, gen, y_target, sampling_ratio, evade=True):
        self.detector = detector
        self.target_layer = target_layer
        self.layer_mismatch = (target_layer != detector.target_layer)
        self.model_latent = get_latent(model, target_layer) if self.layer_mismatch else detector.model_latent
        if self.layer_mismatch:
            # re-extract latent to keep sample latent consistent in different layers
            gen.shuffle = False
            gen.reset()
            self.latent_clean, label_clean, self.l2_norm_clean, acc, acc_target = extract_activation_with_gen(
                self.model_latent, gen, target=y_target)
            latent_detect, label_detect, _, _, _ = extract_activation_with_gen(
                detector.model_latent, gen)
            assert (label_detect == label_clean).mean() == 1.0
            gen.shuffle = True
            gen.reset()
        else:
            self.latent_clean, label_clean, self.l2_norm_clean, acc, acc_target = extract_activation_with_gen(
                self.model_latent, gen, detector.activation_path % "clean_test", target=y_target)
        self.is_target = (label_clean == y_target)
        self.detected = detector.detect_latent(
            latent_detect[self.is_target] if self.layer_mismatch else self.latent_clean[self.is_target])[-1]
        print("clean accuracy: %f, target category acc: %f" % (acc, acc_target))

        # ------------- Preparation phase -------------
        # Step 1: remove outliers with DBSCAN
        X = self.latent_clean[self.is_target] * self.l2_norm_clean[self.is_target, np.newaxis]
        index_outlier = self.remove_outlier(X)

        X = X[~index_outlier]
        self.target_latent = np.mean(X, axis=0)
        self.mse_radius = 0.0
        self.target_direction = normalize(self.target_latent)
        self.target_direction_init = self.target_direction.copy()
        self.cosine_radius = 1.0
        self.latent_clean_target = self.latent_clean[self.is_target][~index_outlier]
        self.l2_norm_clean_target = self.l2_norm_clean[self.is_target][~index_outlier]
        self.detected_clean_target = self.detected[~index_outlier]
        self.latent_undetected = [self.latent_clean_target]
        self.latent_detected = []
        self.boundary_percentile = 50

        # sampling subspaces when sampling ratio is less than 1.
        self.sampling_ratio = sampling_ratio
        self.sampling_masks = None
        self.cosine_radius_sub = None
        self.target_subdirections = None

        self.n_query = 0
        self.n_preparation_generation_success = 0
        self.fpr = detector.fp_percentile

        self.evade = evade
        self.evade_direction_best = None
        if evade:
            self.evade_direction = np.zeros_like(self.target_direction)
            self.evade_threshold = 1.0

    def remove_outlier(self, X, debug=True):
        l2_dist = euclidean_distances(X, X)
        l2_dist.sort(axis=1)
        l2_dist_nn = l2_dist[:, 1]
        l2_dist_nn.sort()
        percentile = 5
        while percentile < 90:
            eps = np.percentile(l2_dist_nn, 100 - percentile)
            dbscan = DBSCAN(eps=eps, min_samples=len(X) * percentile // 100).fit(X)
            index_outlier = (dbscan.labels_ == -1)
            print("there are %d outliers out of %d samples to remove." % (index_outlier.sum(), len(index_outlier)))
            if 100 * index_outlier.sum() / len(index_outlier) > 10:  # until remove more than 10% outliers
                break
            percentile = percentile + 5  # tighter to remove more
        return index_outlier

    def get_radius(self, percentile):
        d_mse = (self.latent_clean_target - self.target_latent) ** 2
        print("benign samples mse percentile (0, 10, 20, ..., 100):\n",
              [np.percentile(d_mse.mean(axis=-1), np.arange(0, 101, 10))])
        self.mse_radius = np.percentile(d_mse.mean(axis=-1), percentile)
        d_mse_all = (self.latent_clean * self.l2_norm_clean[:, np.newaxis] - self.target_latent) ** 2
        mse_radius_all = np.percentile(d_mse_all.mean(axis=-1), percentile)
        print("mse radius of target/all: %.3f/%.3f" % (self.mse_radius, mse_radius_all))
        self.detector.upsample_detect(2.0)
        self.cosine_radius = self.get_cosine_radius(percentile)
        self.get_sub_directions(percentile)

    def get_cosine_radius(self, percentile):
        d_cosine = cosine_similarity(self.latent_clean_target, self.target_direction)
        print("benign samples cosine percentile (0, 10, 20, ..., 100):\n",
              [np.percentile(d_cosine, np.arange(0, 101, 10))])
        cosine_radius = np.percentile(d_cosine, 100 - percentile)
        d_cosine_all = cosine_similarity(self.latent_clean, self.target_direction)
        cosine_radius_all = np.percentile(d_cosine_all, 100 - percentile)
        print("cosine radius of target/all on percentile %.0f: %.2f/%.2f" % (percentile, cosine_radius, cosine_radius_all))
        return cosine_radius

    def get_sub_directions(self, percentile):
        if self.sampling_ratio < 1.:
            self.sampling_masks = np.zeros((int(1/self.sampling_ratio*5), len(self.target_direction)), dtype=np.bool)
            refresh_random_sampling_masks(self.sampling_masks, self.sampling_ratio)
            self.target_subdirections = normalize(self.sampling_masks * self.target_direction)
            latent_clean_target_expanded = self.latent_clean_target[:, np.newaxis, :]
            d_cosine_sub = cosine_similarity(normalize(latent_clean_target_expanded * self.sampling_masks),
                                             self.target_subdirections)
            effective = (np.percentile(d_cosine_sub, 99, axis=0) > 0)  # ensure that neurons in subspace are activated
            self.sampling_masks = self.sampling_masks[effective]
            self.target_subdirections = self.target_subdirections[effective]
            self.cosine_radius_sub = np.percentile(d_cosine_sub[:, effective], 100-percentile, axis=0)


class FeatureIndistinguishableAttack:
    def __init__(self, y_target, oracle_detector, model=None, gen=None,
                 sampling_ratio=1.0, debug=False, dataset=None, target_layer=None, evade=True,
                 dynamic_layer_detector=None, **kwargs):
        self.fd = [FeatureDistribution(oracle_detector, model, target_layer, gen, y_target, sampling_ratio, evade=evade)]
        if dynamic_layer_detector is not None:
            self.fd.append(FeatureDistribution(
                dynamic_layer_detector, model, dynamic_layer_detector.target_layer, gen, y_target, sampling_ratio, evade=evade))  # +early_layer False
        self.target = y_target
        self.oracle = oracle_detector
        self.model = model
        self.gen = gen
        self.debug = debug
        self.dataset = dataset
        self.attack_step = None

    def generate_np(self, x_eval, eps, eps_iter, nb_iter, clip_min, clip_max, recall=True,
                    query_batch_size=32, ce=False, nb_round=1, **kwargs):
        assert x_eval.min() >= clip_min
        assert x_eval.max() <= clip_max

        def iterative_optimize(x, recall_iter=recall):
            x_adv = x.copy()
            x_adv_best = x.copy()
            n_detector = len(self.fd)
            loss_evade_best = np.ones((n_detector, len(x))) * np.inf
            loss_cosine_old = np.ones((n_detector, len(x))) * -np.inf
            coeff = np.ones((n_detector, len(x))) * 0.1  # to control the trad-off between increasing cosine and evading
            coeff_scale_up = 1.2
            coeff_scale_down = coeff_scale_up ** 1.5

            for i in range(nb_iter):
                x_adv_uncut, result_tuple = self.attack_step.generate_np(x_adv, coeff)
                info_debug_list = []
                success_global = np.ones(len(x), dtype=bool)
                cosine_better_global = np.ones(len(x), dtype=bool)
                evade_better_global = np.ones(len(x), dtype=bool)
                for j, (r, fd) in enumerate(zip(result_tuple, self.fd)):
                    if self.debug:
                        cosine_oracle = r.pop()
                    if fd.sampling_masks is not None:
                        loss_cosine_sub = r.pop()
                    if fd.evade:
                        loss_evade, below_threshold = r.pop()
                    predict_adv, loss_mse, loss_cosine, reach_target, within_cos_radius, within_mse_radius = r

                    if fd.sampling_masks is not None:
                        within_cos_radius &= ((loss_cosine_sub > fd.cosine_radius_sub).sum(axis=1) > len(
                            fd.sampling_masks) * 0.9)

                    index = within_cos_radius & reach_target
                    success_global &= index
                    if fd.evade:
                        cosine_better = loss_cosine > loss_cosine_old[j]
                        cosine_better_global &= cosine_better
                        loss_evade_better = np.less_equal(loss_evade, loss_evade_best[j])
                        evade_better_global &= loss_evade_better
                        index &= loss_evade_better
                        loss_evade_best[j][index] = loss_evade[index]
                    loss_cosine_old[j] = loss_cosine

                    if i % 100 == 0:
                        info_debug = "[%s] acc %.2f, mse loss %.3f (%.3f, %.3f), cosine similarity %.3f (%.3f, %.3f), coeff %.3f" % (
                            fd.target_layer, reach_target.mean(), loss_mse.mean(), fd.mse_radius, within_mse_radius.mean(),
                            loss_cosine.mean(), fd.cosine_radius, within_cos_radius.mean(), coeff[j].mean()
                        )
                        if fd.evade:
                            info_debug += ", evade loss %.3f (%.3f, %.3f)" % (
                                loss_evade.mean(), fd.evade_threshold, below_threshold.mean())
                        if self.debug:
                            thres = fd.detector.thresholds[fd.detector.is_effective]
                            info_debug += ', oracle %s' % ', '.join(
                                ["%.3f (%.3f, %.3f)" % (cos, thres, neg_ratio) for cos, thres, neg_ratio in zip(
                                    cosine_oracle.mean(axis=0), thres, (cosine_oracle < thres[np.newaxis,]).mean(axis=0))])
                        if fd.sampling_masks is not None:
                            info_debug += ', sub cosine %s' % ', '.join(
                                ["%.3f (%.3f)" % (cos, r_cos) for cos, r_cos in zip(
                                    loss_cosine_sub.mean(axis=0), fd.cosine_radius_sub)])
                        info_debug_list.append(info_debug)

                index_better = success_global & cosine_better_global
                for j, fd in enumerate(self.fd):
                    if fd.evade:
                        coeff[j][index_better & (coeff[j] < 1e3)] *= coeff_scale_down
                        coeff[j][~index_better & (coeff[j] > 1e-3)] /= coeff_scale_up
                index = success_global & evade_better_global
                x_adv_best[index] = x_adv[index]

                if i % 100 == 0:
                    print("round %5d: %s" % (i, ' | '.join(info_debug_list)))
                x_adv = x + (x_adv_uncut - x).clip(-eps, eps)
            return x_adv_best if recall_iter else x_adv

        def run_preparation_and_calculate_radius(n_round, recall_round):
            x_ori, _ = keep_correct_and_no_target(self.model, self.gen, query_batch_size, self.target)
            x_adv = iterative_optimize(x_ori, recall_round)
            detected = np.zeros(len(x_adv), dtype=np.bool)
            latent_list = []
            for fd in self.fd:
                fd.detector.refresh_random_state()
                fd.detector.upsample_detect(0)
                (_, _, detected_round), (latent, predict) = fd.detector.detect_input(x_adv)
                reach_target = (predict.argmax(axis=-1) == self.target)
                fd.n_query += reach_target.sum()
                fd.n_preparation_generation_success += reach_target.sum()
                detected |= detected_round
                if fd.layer_mismatch:
                    latent, _, _ = fd.model_latent.predict_on_batch(x_adv)
                latent_list.append(latent)

            for fd, latent in zip(self.fd, latent_list):
                index = reach_target & detected
                if index.sum() > 0:
                    fd.latent_detected.append(latent[index])
                index = reach_target & ~detected
                if index.sum() > 0:
                    fd.latent_undetected.append(latent[index])

            attack_success = reach_target & (~detected)
            print("preparation round %d: [out of %d] %d  attack success, %d reach target, %d get detected" % (
                n_round, len(x_adv), attack_success.sum(), reach_target.sum(), detected.sum()
            ))
            reset_cosine_radius(n_round, latent_list, reach_target, detected)
            return detected[reach_target].mean(), reach_target, detected, latent_list

        def reset_cosine_radius(n_round, latent_list, reach_target, detected):
            for i, (fd, latent) in enumerate(zip(self.fd, latent_list)):
                reset_cosine_radius_fp(i, fd, latent, n_round, reach_target, detected)

        def reset_cosine_radius_fp(i, fd, latent, n_round, reach_target, detected):
            # set radius to ensure all reach target
            n_reach_target = reach_target.sum()
            cos = cosine_similarity(latent, fd.target_direction)
            cosine_radius_old = fd.cosine_radius
            if n_reach_target > 0 and n_round != 0:
                # except first round since without evade loss cosine may be much higher (since not bounded)
                fd.cosine_radius = max(fd.cosine_radius, cos[reach_target].min())
            if (len(reach_target) - n_reach_target) > 0:
                fd.cosine_radius = max(fd.cosine_radius, cos[~reach_target].max())
            if fd.cosine_radius != cosine_radius_old:
                print("cosine radius is set to %.2f" % fd.cosine_radius)
            K.set_value(self.attack_step.cosine_radius[i], fd.cosine_radius)

        if self.attack_step is None:  # initialize computation graph and perform preparation adjustment
            # ------------- Preparation phase -------------
            # Step 2: run preparation rounds and determine the centroid and radius
            for fd in self.fd:
                fd.get_radius(100)

            attack_params = {
                "eps_iter": eps_iter,
                "clip_min": clip_min,
                "clip_max": clip_max,
                "y_target": self.target,
                "ce": ce
            }
            self.attack_step = FeatureSpaceAttackStep(self.fd, debug=self.debug, **attack_params)

            for i in range(nb_round):
                positive_ratio, reach_target, detected, latent_list = run_preparation_and_calculate_radius(i, True)  # i > 0

                if positive_ratio == 0.0:  # no positive samples for evade direction estimation
                    break
                if i == 0 and positive_ratio == 1.0:
                    # check if detector's false positive is too high to estimate centroid with all clean target samples
                    # by randomly get one negative sample and query in similarity order to get 10 total
                    fd = self.fd[0]
                    index_shuffle = np.arange(len(fd.latent_clean_target))
                    np.random.shuffle(index_shuffle)
                    index_first_negative = np.argwhere(~fd.detected_clean_target[index_shuffle]).squeeze(1)[0]
                    index_queried = index_shuffle[:index_first_negative+1]
                    similarity_to_first_negative = cosine_similarity(fd.latent_clean_target,
                                                                     fd.latent_clean_target[index_shuffle[index_first_negative]])
                    query_order = similarity_to_first_negative.argsort()[::-1]
                    num_negative_to_get = min(10, (~fd.detected_clean_target).sum())
                    detected_in_query_order = fd.detected_clean_target[query_order]
                    index_queried = np.append(index_queried,
                                              query_order[:(np.argwhere(~detected_in_query_order).squeeze(1)[num_negative_to_get - 1] + 1)])
                    index_queried = np.unique(index_queried)
                    latent_queried = fd.latent_clean_target[index_queried]
                    l2_norm_queired = fd.l2_norm_clean_target[index_queried]
                    detected_queried = fd.detected_clean_target[index_queried]
                    fd.n_query += len(index_queried)

                    index_untested = ~np.isin(np.arange(len(fd.latent_clean_target)), index_queried)
                    latent_untested = fd.latent_clean_target[index_untested]
                    l2_norm_untested = fd.l2_norm_clean_target[index_untested]
                    fd.latent_undetected = [latent_queried[~detected_queried], latent_untested]
                    fd.latent_detected = [latent_queried[detected_queried]] + fd.latent_detected

                    if detected_queried.mean() > 0.5 or fd.evade is False:
                        print("false positive rate is too high, reset centroid with weighted average.")
                        fd.latent_undetected.pop()
                        fd.latent_clean_target = latent_queried[~detected_queried]
                        fd.l2_norm_clean_target = l2_norm_queired[~detected_queried]
                        # weighted average with 2.0 weight on tested negative and 1.0 weight on untested samples
                        fd.target_latent = np.average(
                            np.concatenate([fd.latent_clean_target * fd.l2_norm_clean_target[:, np.newaxis],
                                            latent_untested * l2_norm_untested], axis=0),
                            weights=[2.0] * len(fd.latent_clean_target) + [1.0] * len(latent_untested), axis=0)
                        fd.target_direction = normalize(fd.target_latent)
                        fd.target_direction_init = fd.target_direction.copy()
                        fd.get_radius(100)
                        K.set_value(self.attack_step.target_latent[0], fd.target_latent)
                        K.set_value(self.attack_step.target_direction[0], fd.target_direction)
                        K.set_value(self.attack_step.mse_radius[0], fd.mse_radius)
                        reset_cosine_radius_fp(0, fd, latent_list[0], i, reach_target, detected)

                for j, fd in enumerate(self.fd):
                    # get the best distinguishable evade direction
                    detected_dirs = np.concatenate(fd.latent_detected, axis=0) if len(fd.latent_detected) else np.array([])
                    undetected_dirs = np.concatenate(fd.latent_undetected, axis=0)
                    if len(detected_dirs) > 0:
                        evade_dirs = [detected_dirs.mean(axis=0)]  # global
                        if len(fd.latent_detected) > 1:
                            evade_dirs.append(fd.latent_detected[-1].mean(axis=0))  # local
                        if fd.evade_direction_best is not None:
                            evade_dirs.append(fd.evade_direction_best)
                        evade_dirs = decomposition_np(np.stack(evade_dirs), fd.target_direction_init)
                        gap_evade_dirs = eval_direction(evade_dirs, detected_dirs, undetected_dirs)
                        fd.evade_direction_best = evade_dirs[gap_evade_dirs.argmax()]

                        if fd.evade:
                            fd.evade_direction = fd.evade_direction_best
                            fd.evade_threshold = min(cosine_similarity(detected_dirs, fd.evade_direction_best).min(),
                                                     cosine_similarity(undetected_dirs, fd.evade_direction_best).max())

                            K.set_value(self.attack_step.evade_direction[j], fd.evade_direction_best)
                            K.set_value(self.attack_step.evade_threshold[j], fd.evade_threshold)

                        print("update centroid with positive samples!")
                        coeff = 0.1 if self.dataset != "youtube_face" else 1e-3
                        new_target_direction = normalize(
                            fd.target_direction_init + coeff * positive_ratio * normalize(fd.latent_clean_target.mean(axis=0) - fd.evade_direction_best))
                        fd.target_direction = new_target_direction
                        fd.get_radius(100)
                        K.set_value(self.attack_step.target_direction[j], fd.target_direction)
                        reset_cosine_radius_fp(j, fd, latent_list[j], i, reach_target, detected)

                        undetected_scores = cosine_similarity(fd.target_direction, fd.latent_undetected[0])
                        detected_scores = cosine_similarity(fd.target_direction, detected_dirs)
                        cosine_radius_old = fd.cosine_radius
                        fd.cosine_radius = max(fd.cosine_radius,
                                               min(np.percentile(undetected_scores, fd.boundary_percentile),
                                                   max(np.percentile(undetected_scores, 10),
                                                       np.percentile(detected_scores, 90))))
                        if fd.cosine_radius != cosine_radius_old:
                            print("cosine radius is adjusted from %.2f to %.2f" % (cosine_radius_old, fd.cosine_radius))
                            K.set_value(self.attack_step.cosine_radius[j], fd.cosine_radius)

                if i > 0 and positive_ratio < 0.05:  # if positive ratio is satisfied, early stop preparation stage
                    break
        return iterative_optimize(x_eval)
