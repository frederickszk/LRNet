import argparse

import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
from data_utils import *
from os.path import join


def load_data_train(add_real, add_fake, block_size):
    train_samples, train_samples_diff, train_labels = get_data(join(add_real, "train/"), 0, block_size)
    tmp_samples, tmp_samples_diff, tmp_labels = get_data(join(add_fake, "train/"), 1, block_size)

    train_samples = np.concatenate((train_samples, tmp_samples), axis=0)
    train_samples_diff = np.concatenate((train_samples_diff, tmp_samples_diff), axis=0)
    train_labels = np.concatenate((train_labels, tmp_labels), axis=0)

    # We need to copy this labels to _diff for we need to shuffle them separately
    train_labels_diff = train_labels.copy()

    """
    Shuffle the training data
    """
    np.random.seed(200)
    np.random.shuffle(train_samples)
    np.random.seed(200)
    np.random.shuffle(train_labels)

    np.random.seed(500)
    np.random.shuffle(train_samples_diff)
    np.random.seed(500)
    np.random.shuffle(train_labels_diff)

    """
    Flush the memory
    """
    tmp_samples = []
    tmp_samples_diff = []
    tmp_labels = []

    return train_samples, train_samples_diff, train_labels, train_labels_diff


def load_data_test(add_real, add_fake, block_size):
    test_samples, test_samples_diff, test_labels, test_labels_video, test_sv, test_vc = \
        get_data_for_test(join(add_real, "test/"), 0, block_size)
    tmp_samples, tmp_samples_diff, tmp_labels, tmp_labels_video, tmp_sv, tmp_vc = \
        get_data_for_test(join(add_fake + "test/"), 1, block_size)

    test_samples = np.concatenate((test_samples, tmp_samples), axis=0)
    test_samples_diff = np.concatenate((test_samples_diff, tmp_samples_diff), axis=0)
    test_labels = np.concatenate((test_labels, tmp_labels), axis=0)
    test_labels_video = np.concatenate((test_labels_video, tmp_labels_video), axis=0)
    test_sv = np.concatenate((test_sv, tmp_sv), axis=0)

    test_vc.update(tmp_vc)

    """
    Flush the memory
    """
    tmp_samples = []
    tmp_samples_diff = []
    tmp_labels = []

    return test_samples, test_samples_diff, test_labels, test_labels_video, test_sv, test_vc


def main(args):
    if_train = args.train
    if_evaluate = args.evaluate
    if_gpu = args.gpu

    """
    Initialization
    """
    BLOCK_SIZE = 60
    DROPOUT_RATE = 0.5
    RNN_UNIT = 64

    add_real = './datasets/Origin/c23/'
    add_fake = './datasets/DF/c23/'
    add_weights = './weights/tf/'

    if if_gpu:
        # Optional to uncomment if some bugs occur.
        # os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        gpus = tf.config.list_physical_devices(device_type='GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(device=gpu, enable=True)
        device = "CPU" if len(gpus) == 0 else "GPU"
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        device = 'CPU'

    """
    Logs
    """
    print("=======================")
    print("==        LOG        ==")
    print("=======================")
    print("\n")
    print("#-------Status--------#")
    print("If train a new model: ", if_train)
    print("If evaluate the model: ", if_evaluate)
    print("Using device: ", device)
    print("#-----Status End------#")
    print("\n")
    print("#-----Parameters------#")
    print("Block size (frames per sample): ", BLOCK_SIZE)
    print("RNN hidden units: ", RNN_UNIT)
    print("Dataset of real samples: ", add_real)
    print("Dataset of fake samples: ", add_fake)
    print("Folder of model weights: ", add_weights)
    print("#---Parameters End----#")
    print("\n")
    print("=======================")
    print("==      LOG END      ==")
    print("=======================")

    """
    Define the models.
    # model = g1, model_diff = g2
    """
    model = K.Sequential([
        layers.InputLayer(input_shape=(BLOCK_SIZE, 136)),
        layers.Dropout(0.25),
        layers.Bidirectional(layers.GRU(RNN_UNIT)),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(64, activation='relu'),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(2, activation='softmax')
    ])
    model_diff = K.Sequential([
        layers.InputLayer(input_shape=(BLOCK_SIZE - 1, 136)),
        layers.Dropout(0.25),
        layers.Bidirectional(layers.GRU(RNN_UNIT)),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(64, activation='relu'),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(2, activation='softmax')
    ])

    lossFunction = K.losses.SparseCategoricalCrossentropy(from_logits=False)
    optimizer = K.optimizers.Adam(learning_rate=0.001)

    if if_train:
        train_samples, train_samples_diff, train_labels, train_labels_diff = \
            load_data_train(add_real, add_fake, BLOCK_SIZE)
        test_samples, test_samples_diff, test_labels, test_labels_video, test_sv, test_vc = \
            load_data_test(add_real, add_fake, BLOCK_SIZE)

        # ----For g1----#
        callbacks = [
            K.callbacks.ModelCheckpoint(
                filepath=add_weights + 'g1.h5',
                save_best_only=True,
                monitor='val_accuracy',
                verbose=1)
        ]
        model.compile(optimizer=optimizer, loss=lossFunction, metrics=['accuracy'])
        model.fit(train_samples, train_labels, batch_size=1024,
                  validation_data=(test_samples, test_labels), epochs=500,
                  shuffle=True, callbacks=callbacks)
        # ----g1 end----#

        # ----For g2----#
        callbacks_diff = [
            K.callbacks.ModelCheckpoint(
                filepath=add_weights + 'g2.h5',
                save_best_only=True,
                monitor='val_accuracy',
                verbose=1)
        ]
        model_diff.compile(optimizer=optimizer, loss=lossFunction, metrics=['accuracy'])
        model_diff.fit(train_samples_diff, train_labels_diff, batch_size=1024,
                       validation_data=(test_samples_diff, test_labels), epochs=400,
                       shuffle=True, callbacks=callbacks_diff)
        # ----g2 end----#

    if if_evaluate:
        if not if_train:
            # Test data have been loaded if trained. Thus if not trained we should load test data here.
            test_samples, test_samples_diff, test_labels, test_labels_video, test_sv, test_vc = \
                load_data_test(add_real, add_fake, BLOCK_SIZE)

        # ----For g1----#
        model.compile(optimizer=optimizer, loss=lossFunction, metrics=['accuracy'])
        model.load_weights(add_weights + 'g1.h5')
        loss_g1, acc_g1 = model.evaluate(test_samples, test_labels, batch_size=512)
        # ----g1 end----#

        # ----For g2----#
        model_diff.compile(optimizer=optimizer, loss=lossFunction, metrics=['accuracy'])
        model_diff.load_weights(add_weights + 'g2.h5')
        loss_g2, acc_g2 = model_diff.evaluate(test_samples_diff, test_labels, batch_size=512)
        # ----g2 end----#

        """
        Evaluate the merged prediction (sample-level and video-level)
        """
        # ----Sample-level----#
        prediction = model.predict(test_samples)
        prediction_diff = model_diff.predict(test_samples_diff)
        count_s = 0
        total_s = test_labels.shape[0]
        mix_predict = []
        for i in range(len(prediction)):
            mix = prediction[i][1] + prediction_diff[i][1]
            if mix >= 1:
                result = 1
            else:
                result = 0
            if result == test_labels[i]:
                count_s += 1
            mix_predict.append(mix / 2)

        # ----Video-level----#
        prediction_video = merge_video_prediction(mix_predict, test_sv, test_vc)
        count_v = 0
        total_v = len(test_labels_video)
        for i, pd in enumerate(prediction_video):
            if pd >= 0.5:
                result = 1
            else:
                result = 0
            if result == test_labels_video[i]:
                count_v += 1

        print("\n")
        print("#----Evaluation  Results----#")
        print("Evaluation (g1) - Acc: {:.4}, Loss: {:.4}".format(acc_g1, loss_g1))
        print("Evaluation (g2) - Acc: {:.4}, Loss: {:.4}".format(acc_g2, loss_g2))
        print("Accuracy (sample-level): ", count_s / total_s)
        print("Accuracy (video-level): ", count_v / total_v)
        print("#------------End------------#")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Training and evaluating of LRNet (Tensorflow 2.x version).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-t', '--train', action='store_true',
                        help="If train the model."
                        )
    parser.add_argument('-e', '--evaluate', action='store_true',
                        help="If evaluate the model."
                        )
    parser.add_argument('-g', '--gpu', action='store_true',
                        help="If use the GPU(CUDA) for training."
                        )
    args = parser.parse_args()
    main(args)
