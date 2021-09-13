import random
import numpy as np
# import cupy as cp
import sklearn.preprocessing
import tensorflow as tf
import keras.backend as K
from keras.utils import to_categorical
import keras
from keras.models import Model
from keras.layers import Flatten, Lambda, Reshape  # , LayerNormalization
from keras.preprocessing.image import ImageDataGenerator, NumpyArrayIterator, DirectoryIterator
from tensorflow.keras.datasets import cifar10, mnist
import pandas as pd
import os
from tqdm import tqdm
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler


def fix_random_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def fix_gpu_memory(mem_fraction=1):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_fraction)
    tf_config = tf.ConfigProto(gpu_options=gpu_options)
    tf_config.gpu_options.allow_growth = True
    tf_config.log_device_placement = False
    tf_config.allow_soft_placement = True
    init_op = tf.global_variables_initializer()
    sess = tf.Session(config=tf_config)
    sess.run(init_op)
    K.set_session(sess)
    return sess


def concatenate_dict_as_string(dct):
    dct_copy = dct.copy()
    if "target_list" in dct_copy.keys() and len(dct_copy["target_list"]) > 100:
        dct_copy["target_list"] = "all_label"
    return "_".join(["%s=%s" % (key, value) for key, value in dct_copy.items()])


def get_model_by_name(dataset_name):
    if dataset_name == 'cifar10':
        from keras_model.cifar10 import resnet_v1
        model = resnet_v1((32, 32, 3), 3 * 6 + 2, mean=np.load("cifar10_train_per_pixel_channel_mean.npy"))
    elif dataset_name == 'mnist':
        from keras_model.mnist import get_mnist_model
        model = get_mnist_model()
    else:
        raise NotImplementedError
    return model


def load_data(dataset_info, batch_size=32, shuffle=True):
    datagen_aug = ImageDataGenerator(rescale=1 / 255.,
                                     width_shift_range=0.1,  # randomly shift images horizontally
                                     height_shift_range=0.1,  # randomly shift images vertically
                                     horizontal_flip=True  # randomly flip images
                                     )
    datagen = ImageDataGenerator(rescale=1 / 255.)
    dataset = dataset_info["dataset"]
    common_params = {
        "batch_size": batch_size,
        "shuffle": shuffle
    }

    def label_to_one_hot(y):
        return to_categorical(y, num_classes=dataset_info["num_classes"])

    if dataset_info["load_with_keras"]:
        print("load data with keras.datasets")
        entry = {
            "cifar10": cifar10,
            "mnist": mnist
        }
        (x_train, y_train), (x_test, y_test) = entry[dataset].load_data()
        if len(x_test.shape) == 3:
            x_train = x_train[:, :, :, np.newaxis]
            x_test = x_test[:, :, :, np.newaxis]

        y_train = label_to_one_hot(y_train)
        y_test = label_to_one_hot(y_test)

        # show example here
        if "data_augmentation" in dataset_info.keys() and dataset_info["data_augmentation"]:
            generator_train = datagen_aug.flow(x_train, y_train, **common_params)
        else:
            generator_train = datagen.flow(x_train, y_train, **common_params)
        generator_test = datagen.flow(x_test, y_test, **common_params)

        return generator_train, generator_test
    else:
        raise NotImplementedError


class DataGenerator(object):
    def __init__(self, trigger_list, num_classes):
        n_trigger = len(trigger_list)
        self.trigger_index = np.arange(n_trigger)
        self.target_array = np.zeros(n_trigger)
        self.mask_array = np.zeros((n_trigger,) + trigger_list[0][1].shape)
        self.pattern_array = np.zeros((n_trigger,) + trigger_list[0][2].shape)
        for i, trigger in enumerate(trigger_list):
            self.target_array[i], self.mask_array[i], self.pattern_array[i] = trigger
        self.num_classes = num_classes

    def generate_data(self, gen, inject_ratio):
        while 1:
            clean_x, clean_y = next(gen)
            y = clean_y.copy()

            # sample and inject trapdoor as batch
            mask_batch = np.zeros_like(clean_x)
            pattern_batch = np.zeros_like(clean_x)

            inject_prob = np.random.uniform(0, 1, len(clean_x))
            index_inject = (inject_prob < inject_ratio)
            index_trigger_sampled = np.random.choice(self.trigger_index, index_inject.sum())
            mask_batch[index_inject] = self.mask_array[index_trigger_sampled]
            pattern_batch[index_inject] = self.pattern_array[index_trigger_sampled]

            y[index_inject] = to_categorical(self.target_array[index_trigger_sampled], num_classes=self.num_classes)
            x = mask_batch * pattern_batch + (1. - mask_batch) * clean_x

            yield x, y


class TrapdoorModelEvaluationCallback(keras.callbacks.Callback):
    def __init__(self, gen_test, gen_test_adv, model_path, accept_clean_acc, accept_trapdoor_acc):
        self.gen_test = gen_test
        self.gen_test_adv = gen_test_adv
        self.model_path = model_path
        self.best_trapdoor_acc = 0
        self.best_clean_acc = 0
        self.accept_clean_acc = accept_clean_acc
        self.accept_trapdoor_acc = accept_trapdoor_acc

    def on_epoch_end(self, epoch, verbose=0):
        _, clean_acc = self.model.evaluate_generator(self.gen_test, verbose=verbose)
        _, trapdoor_acc = self.model.evaluate_generator(self.gen_test_adv, steps=100, verbose=verbose)

        print("Epoch: {} - Clean Acc {:.4f} - Trapdoor Acc {:.4f}".format(epoch, clean_acc, trapdoor_acc))
        if clean_acc > self.accept_clean_acc and trapdoor_acc > self.accept_trapdoor_acc:
            if clean_acc > self.best_clean_acc:
                print("saving model to %s" % self.model_path)
                self.model.save(self.model_path)
                self.best_clean_acc = clean_acc
                self.best_trapdoor_acc = trapdoor_acc
            elif trapdoor_acc > self.best_trapdoor_acc:
                print("saving model to %s" % self.model_path)
                self.model.save(self.model_path)
                self.best_trapdoor_acc = trapdoor_acc


def get_model_latent(dataset, model_path, target_layer):
    model = get_model_by_name(dataset)
    print("loading model from %s" % model_path)
    model.load_weights(model_path)
    return model, get_latent(model, target_layer)


def get_latent(model, target_layer):
    # prepare the latent extractor model
    latent_output = model.get_layer(target_layer).output
    if len(latent_output.shape) > 2:
        latent_output = Flatten()(latent_output)
    l2_norm = Lambda(lambda x: K.sqrt(K.sum(x ** 2, axis=-1)))(latent_output)
    latent_output_normalized = Lambda(lambda x: x[0] / (x[1] + 1e-5))([latent_output, Reshape((1,))(l2_norm)])
    model_latent = Model(model.input, [latent_output_normalized, model.output, l2_norm])
    return model_latent


def keep_correct_and_no_target(model, gen_test, num_sample_eval, target):
    gen_test.shuffle = True
    x_eval_list = []
    y_eval_list = []
    num_eval = 0
    for _ in range(len(gen_test)):
        x, y = next(gen_test)
        not_effective = ((x.max(axis=(1, 2, 3)) > 1.0) | (x.min(axis=(1, 2, 3)) < 0.0))
        if not_effective.sum() > 0:
            print("there are %d bad samples!" % not_effective.sum())
        # if x.min() < 0.0 or x.max() > 1.0:  # skip bad samples
        #     continue
        predict = model.predict_on_batch(x).argmax(axis=-1)
        label = y.argmax(axis=-1)
        index_keep = (predict == label) & (label != target) & (~not_effective)
        x_eval_list.append(x[index_keep])
        y_eval_list.append(y.argmax(axis=-1)[index_keep])
        num_eval += index_keep.sum()
        if num_eval > num_sample_eval:
            break
    return np.concatenate(x_eval_list, axis=0)[:num_sample_eval], np.concatenate(y_eval_list, axis=0)[:num_sample_eval]


def split_directory_iterator_with_condition(gen, condition):
    classes = gen.classes[condition]
    filenames = [gen.filenames[i] for i in np.argwhere(condition).squeeze()]
    filepaths = [gen._filepaths[i] for i in np.argwhere(condition).squeeze()]
    set_directory_iterator(gen, (classes, filenames, filepaths))


def set_directory_iterator(gen, splits):
    gen.classes, gen.filenames, gen._filepaths = splits
    gen.samples = len(gen.classes)
    gen.n = gen.samples
    gen._set_index_array()
    gen.reset()


def extract_activation_with_gen(model_get_latent, eval_datagen, save_path=None, steps=None, target=None, n=None,
                                save=True, load=True, keep_correct=True):
    if n is None:
        n = eval_datagen.n
    if load and save_path is not None and os.path.exists(save_path):
        latent = np.load(save_path)
        predict = np.load(save_path.replace('.npy', '_label.npy'))
        l2_norm = np.load(save_path.replace('.npy', '_l2_norm.npy'))
        acc = len(predict) / n
        if target is None:
            target_acc = 0
        else:
            if type(eval_datagen) == NumpyArrayIterator:
                target_acc = (predict == target).sum() / (eval_datagen.y.argmax(axis=-1) == target).sum()
            elif type(eval_datagen) == DirectoryIterator:
                target_acc = (predict == target).sum() / (eval_datagen.labels == target).sum()
            else:
                target_acc = 0
    else:
        steps = len(eval_datagen) if steps is None else steps
        if type(eval_datagen) in [NumpyArrayIterator, DirectoryIterator]:
            ground_truth = eval_datagen.y.argmax(axis=-1) if type(
                eval_datagen) is NumpyArrayIterator else eval_datagen.labels
            eval_datagen._set_index_array()
            eval_datagen.reset()
            ground_truth = ground_truth[eval_datagen.index_array]
        else:
            ground_truth = np.ones(n, dtype=np.int) * target
        latent, predict, l2_norm = model_get_latent.predict_generator(eval_datagen, verbose=1, steps=steps)
        predict = predict.argmax(axis=-1)
        # only keep predict correct and non-target ones
        classify_correct = (predict == ground_truth)
        if keep_correct:
            latent = latent[classify_correct]
            predict = predict[classify_correct]
            l2_norm = l2_norm[classify_correct]
        if save_path is not None and save:
            np.save(save_path, latent)
            np.save(save_path.replace('.npy', '_label.npy'), predict)
            np.save(save_path.replace('.npy', '_l2_norm.npy'), l2_norm)
        acc = classify_correct.mean()
        target_acc = 0 if target is None else classify_correct[ground_truth == target].mean()

    not_effective = (latent.max(axis=1) > 1e3)  # remove those abnormally activated samples (keras bug)
    if not_effective.sum() > 0:
        latent = latent[~not_effective]
        predict = predict[~not_effective]
        l2_norm = l2_norm[~not_effective]
    return latent, predict, l2_norm, acc, target_acc


def cosine_similarity(a, b):
    # calculate cosine similarity between given normalized vector
    if len(a.shape) > len(b.shape):
        return (a * b).sum(axis=-1)
    else:
        return (b * a).sum(axis=-1)


def normalize(a):
    return a / np.expand_dims(np.linalg.norm(a, axis=-1) + 1e-5, axis=-1)  # to avoid dividing by 0


def orthogonal_decomposition(a, b):
    projection_on_b = np.dot(a, b) / np.square(b).sum() * b
    orthogonal_a = normalize(a - projection_on_b)
    return orthogonal_a, projection_on_b


def get_trapdoor_gen(gen, target, mask, pattern):
    if type(gen) == NumpyArrayIterator:
        is_not_target = (gen.y.argmax(axis=-1) != target)
        x = mask * pattern * 255. + (1 - mask) * gen.x[is_not_target]
        y = to_categorical(target, gen.y.shape[1])[np.newaxis, :].repeat(len(x), axis=0)
        return ImageDataGenerator(rescale=1 / 255.).flow(x, y, batch_size=gen.batch_size), None
    else:
        adv_gen = DataGenerator([[target, mask, pattern]], gen.num_classes)
        return adv_gen.generate_data(gen, 1.), len(gen)


def train(training_setting, clean_model_path, gen_train, gen_test, trigger_dict, dataset_info, injection_ratio,
          model_path, model=None):
    if model is None:
        model = get_model_by_name(dataset_info["dataset"])
    model.compile(loss='categorical_crossentropy', optimizer=Adam(1e-3), metrics=['accuracy'])
    n_epoch = training_setting["epochs"]
    while not os.path.exists(model_path):
        def lr_schedule(epoch):
            lr = 1e-3
            if epoch > int(n_epoch * 0.6):
                lr *= 1e-1
            elif epoch > int(n_epoch * 0.8):
                lr *= 1e-2
            print('Learning rate: ', lr)
            return lr

        lr_scheduler = LearningRateScheduler(lr_schedule)
        if len(clean_model_path):
            print("train from a clean model")
            if not os.path.exists(clean_model_path):
                save_best = ModelCheckpoint(clean_model_path, save_best_only=True, monitor='val_accuracy', mode='max',
                                            verbose=1)
                model.fit_generator(gen_train, epochs=n_epoch, callbacks=[save_best, lr_scheduler],
                                    validation_data=gen_test, validation_steps=max(300, len(gen_test)))
            model.load_weights(clean_model_path)
        # inject trapdoor on the fly
        trigger_list = []
        for target, triggers in trigger_dict.items():
            for trigger in triggers:
                trigger_list.append((target, trigger["mask"], trigger["pattern"]))
        adv_gen = DataGenerator(trigger_list, dataset_info["num_classes"])
        save_best = TrapdoorModelEvaluationCallback(gen_test, adv_gen.generate_data(gen_test, 1.), model_path,
                                                    training_setting["accept_clean_acc"],
                                                    training_setting["accept_trapdoor_acc"])
        from keras.callbacks import CSVLogger
        csv_logger = CSVLogger('log.csv', append=True, separator=',')
        model.fit_generator(adv_gen.generate_data(gen_train, injection_ratio), epochs=n_epoch,
                            steps_per_epoch=int(len(gen_train) * (1 + injection_ratio)),
                            callbacks=[save_best, lr_scheduler, csv_logger])
        if not os.path.exists(model_path):
            n_epoch *= 2
            print("no good model exists in %s, double the training epochs!" % model_path)

    K.clear_session()


def refresh_random_sampling_masks(masks, sampling_ratio):
    masks[:, :] = 0
    len_mask = masks.shape[1]
    index = np.arange(len_mask)
    n_keep = int(len_mask * sampling_ratio)
    for i in range(masks.shape[0]):
        masks[i, np.random.choice(index, n_keep, replace=False)] = 1
