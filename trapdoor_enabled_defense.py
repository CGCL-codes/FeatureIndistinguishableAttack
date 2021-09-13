import pickle
from trapdoor_enabled_detector import TrapdoorEnabledDetector
from attacks.feature_indistinguishable_attack import FeatureIndistinguishableAttack
from cleverhans import attacks
from cleverhans.utils_keras import KerasModelWrapper
from utils import *


def generate_mask(input_shape, transparency=.1, size=6, n_pieces=1, centered=False):
    # 6x6 pixels with k = 0.1 for single label
    mask = np.zeros(input_shape[:-1])
    if n_pieces == 1:
        if centered:
            start_row, start_column = (np.array(input_shape[:-1]) - size) // 2
        else:
            start_row, start_column = np.array(input_shape[:-1]) - size - 2

        mask[start_row: start_row + size, start_column: start_column + size] = transparency
    else:
        for i, start_location in enumerate(
                (np.random.random((n_pieces, 2)) * (np.array(input_shape[:-1]) - size)).astype(np.int)):
            start_row, start_column = start_location
            # print("start row/column of trigger piece %d are %d/%d" % (i, start_row, start_column))
            mask[start_row: start_row + size, start_column: start_column + size] = transparency
    return mask


def generate_pattern(input_shape):
    pattern_per_channel_list = []
    for i, mu_sigma in enumerate(np.random.random((input_shape[-1], 2))):
        mu, sigma = mu_sigma
        # print("mu/sigma of pattern of channel %d are %f/%f" % (i, mu, sigma))
        pattern_per_channel_list.append(
            np.random.normal(loc=mu, scale=sigma, size=input_shape[:-1])[:, :, np.newaxis])
    pattern = np.clip(np.concatenate(pattern_per_channel_list, axis=-1), 0., 1.)
    return pattern  # * 255.


def generate_trigger_dict(input_shape, target_list, n_trapdoor_per_label, transparency, size, n_pieces):
    trigger_dict = {}
    for target in target_list:
        trigger_dict[target] = []
        for i in range(n_trapdoor_per_label):
            trigger_dict[target].append({
                "mask": generate_mask(input_shape, transparency, size, n_pieces)[:, :, np.newaxis],
                "pattern": generate_pattern(input_shape)
            })
    return trigger_dict


def trapdoor_enabled_defense(gen_train, gen_test, task, trapdoor_setting, dataset_info, training_setting,
                             injection_ratio, target_layer, eval_batch_size, attack_setting, clean_model_path='',
                             debug=True, fpr=5, distinct_method="orthogonal", **params):
    print("trapdoor setting: %s" % trapdoor_setting)
    dataset = dataset_info["dataset"]
    subdir_name = concatenate_dict_as_string(trapdoor_setting)  # show basic configs in the directory name
    exp_dir = "exp_result/%s/%s" % (dataset, subdir_name)
    trigger_path = "%s/triggers.pkl" % exp_dir
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    # ------ First prepare the data generator and trigger dict ------
    if os.path.exists(trigger_path):
        print("loading existing triggers")
        with open(trigger_path, 'rb') as f:
            trigger_dict = pickle.load(f)
    else:
        print("generating triggers for each trapdoor(s) of labels to protect")
        trigger_dict = generate_trigger_dict(dataset_info["img_shape"], **trapdoor_setting)
        with open(trigger_path, 'wb') as f:
            pickle.dump(trigger_dict, f)

    # ------ Then prepare the model ------
    print("training setting: %s" % training_setting)
    model_path = "%s/%s.h5" % (exp_dir, concatenate_dict_as_string(training_setting))
    if not os.path.exists(model_path):
        # train the model
        gen_train.batch_size = training_setting["batch_size"]
        gen_train.reset()
        train(training_setting, clean_model_path, gen_train, gen_test, trigger_dict, dataset_info, injection_ratio,
              model_path)

    # ------ Last perform the attacks and defense evaluation ------
    if dataset == "youtube_face":
        keeper = [gen_train.classes, gen_train.filenames, gen_train._filepaths]
        # random sampling a subset from training set to build defense (since whole costs too much time and memory)
        n_sampled = gen_train.n // 5
        index_sampled = np.random.permutation([True] * n_sampled + [False] * (gen_train.n - n_sampled))
        split_directory_iterator_with_condition(gen_train, index_sampled)
    fpr_of_benign_target_list, trapdoor_acc_list = evaluate(
        task, dataset_info, eval_batch_size, gen_test, gen_train, trigger_dict, exp_dir,
        model_path, target_layer, attack_setting, debug, fpr=fpr, distinct_method=distinct_method)
    if dataset == "youtube_face":
        set_directory_iterator(gen_train, keeper)

    return exp_dir, dataset, trigger_dict.keys(), trapdoor_acc_list, fpr_of_benign_target_list


def evaluate(task, dataset_info, eval_batch_size, gen_test, gen_train, trigger_dict, exp_dir,
             model_path, target_layer, attack_setting, debug, fpr=5, distinct_method="orthogonal"):
    num_sample_evaluate = eval_batch_size
    gen_train.batch_size = eval_batch_size
    gen_test.batch_size = eval_batch_size
    dataset = dataset_info["dataset"]
    max_label_evaluate = 100
    if len(trigger_dict) < max_label_evaluate:
        eval_trigger_dict = trigger_dict
    else:  # sample label to evaluate if there are too many labels
        eval_trigger_dict = {}
        for key in np.random.choice(list(trigger_dict.keys()), max_label_evaluate, replace=False):
            eval_trigger_dict[key] = trigger_dict[key]

    fpr_of_benign_target_list, trapdoor_acc_list = [], []
    result_filename = "result/%s_%s_%s.csv" % (dataset, task, distinct_method)
    write_head = False if os.path.exists(result_filename) else True
    with open(result_filename, "a+") as fp:
        if write_head:
            fp.write("attack,target,# detected,# success,# eval, fpr benign target,"
                     " # trapdoor, sampling ratio, # repeat, target layer, distinct method\n")
        # perform evaluation per protected label
        for target, triggers in list(eval_trigger_dict.items()):
            print("----------------start evaluating target label %d----------------" % target)
            model, model_latent = get_model_latent(dataset, model_path, target_layer)

            # defender builds detector using training set
            detector = TrapdoorEnabledDetector(
                dataset, model_latent, target, target_layer, triggers, gen_train, gen_test, exp_dir,
                fp_percentile=fpr, distinct_method=distinct_method, debug=debug)
            fpr_of_benign_target_list.append(detector.united_fpr_benign_target)
            trapdoor_acc_list.append(detector.trapdoor_acc_record)

            x_eval, y_eval = keep_correct_and_no_target(model, gen_test, num_sample_evaluate, target)

            for attack_name, setting in attack_setting.items():
                # perform evaluation per attack
                if attack_name == "FeatureIndistinguishableAttack" and detector.united_fpr_benign_target == 1.0:
                    print("all false positive, fsa can't apply, continue!")
                    K.clear_session()
                    continue
                evaluate_attack(
                    model, attack_name, setting, target, x_eval, eval_batch_size, dataset_info, detector, fp, gen_test,
                    exp_dir, target_layer, debug=debug)
            K.clear_session()
    return fpr_of_benign_target_list, trapdoor_acc_list


def evaluate_attack(model, attack_name, attack_setting, target, x_eval, batch_size, dataset_info,
                    detector, result_fp, gen_test, exp_dir, target_layer, debug=False):
    print("----------------evaluating attack %s----------------" % attack_name)
    print(attack_setting)
    filename = concatenate_dict_as_string(attack_setting)
    adv_save_path = "%s/x_adv_%s_%d_%s.npy" % (exp_dir, attack_name, target, filename)
    if attack_name == "FeatureIndistinguishableAttack":
        adv_save_path = adv_save_path.replace('.npy', '_%s.npy' % detector.distinct_method)
        if "target_layer" in attack_setting.keys() and detector.target_layer != attack_setting["target_layer"]:
            adv_save_path = adv_save_path.replace('.npy', '_detect_layer=%s.npy' % detector.target_layer)
    # -------- First generate adversarial examples --------
    if os.path.exists(adv_save_path):
        x_adv = np.load(adv_save_path)
        attack = None
    else:
        attack_entry = {
            "ProjectedGradientDescent": attacks.ProjectedGradientDescent,
            "CarliniWagnerL2": attacks.CarliniWagnerL2,
            "FeatureIndistinguishableAttack": FeatureIndistinguishableAttack
        }
        assert attack_name in attack_entry.keys()
        attack_params = attack_setting.copy()
        for key in attack_params.keys():
            if key in ['eps', 'eps_iter', 'clip_max']:
                attack_params[key] /= 255.  # normalize attack settings for normalized input space
        if attack_name == "CarliniWagnerL2":
            attack_params['batch_size'] = batch_size
        attack_params['y_target'] = to_categorical(
            target, dataset_info["num_classes"])[np.newaxis, :].repeat(batch_size, axis=0)

        if attack_name != 'FeatureIndistinguishableAttack':
            wrap = KerasModelWrapper(model)
            attack = attack_entry[attack_name](wrap, sess=K.get_session())
        else:
            attack = attack_entry[attack_name](target, detector, **{
                "model": model,
                "gen": gen_test,
                "debug": debug,
                "dataset": dataset_info["dataset"],
                "target_layer": attack_setting["target_layer"] if "target_layer" in attack_setting.keys() else target_layer,
                "evade": attack_setting["evade"] if "evade" in attack_setting.keys() else True
            })

        x_adv = np.zeros_like(x_eval)
        for i in tqdm(range(len(x_eval) // batch_size)):
            x_eval_batch = x_eval[i * batch_size: (i + 1) * batch_size]
            x_adv_batch = attack.generate_np(
                x_eval_batch,
                **attack_params
            )
            x_adv[i * batch_size: (i + 1) * batch_size] = x_adv_batch
        np.save(adv_save_path, x_adv)

    # -------- Then perform trapdoor-enabled trapdoor evaluation --------
    latent_adv = []
    predict_adv = []
    for i in tqdm(range(len(x_adv) // batch_size)):
        x_adv_batch = x_adv[i * batch_size: (i + 1) * batch_size]
        latent_adv_batch, predict_adv_batch, _ = detector.model_latent.predict_on_batch(x_adv_batch)
        latent_adv.append(latent_adv_batch)
        predict_adv.append(predict_adv_batch)
    latent_adv = np.concatenate(latent_adv, axis=0)
    predict_adv = np.concatenate(predict_adv, axis=0)
    label_adv = predict_adv.argmax(axis=-1)
    attack_success = (label_adv == target)

    scores_adv, detected_per_trapdoor, detected_adv = detector.detect_latent(latent_adv[attack_success])
    result_tuple = (detected_adv.sum(), attack_success.sum(), latent_adv.shape[0])
    result_log = "%s,%d,%d,%d,%d,%f,%d,%.2f,%d,%s,%s" % (
            ("%s_%s" % (attack_name, filename), target) + result_tuple + (
        detector.united_fpr_benign_target, len(detector.signatures),
        1.0, 1, target_layer, detector.distinct_method))
    if attack_name == 'FeatureIndistinguishableAttack' and attack is not None:
        result_log += ",%d, %d, %r, %d" % (attack.fd[0].n_query, attack.fd[0].n_preparation_generation_success,
                                           attack_setting["evade"], attack_setting["query_batch_size"])
    if result_fp is not None:
        result_fp.write(result_log+'\n')
        result_fp.flush()
    print(">> total number of detected adversarial/attack success/attacked are %d, %d, %d" % result_tuple)
