import os
from os.path import join
import numpy as np
import torch
import torch.utils.data as Data
from tqdm import tqdm


def get_data(path, fake, block):
    """
    Read the data into memory (for training).
    :param path: The path of the folder containing landmarks files.
    :param fake: Assign the label of the data. Original(real) = 0, and manipulated(fake) = 1.
    :param block: The length of a 'block', i.e., the frames number of a video sample.
    :return:
    """
    files = os.listdir(path)
    x = []
    x_diff = []
    y = []

    for file in tqdm(files):
        vectors = np.loadtxt(join(path, file))
        for i in range(0, vectors.shape[0] - block, block):
            vec = vectors[i:i + block, :]
            x.append(vec)
            vec_next = vectors[i + 1:i + block, :]
            vec_next = np.pad(vec_next, ((0, 1), (0, 0)), 'constant', constant_values=(0, 0))
            vec_diff = (vec_next - vec)[:block - 1, :]
            x_diff.append(vec_diff)
            y.append(fake)
    return np.array(x), np.array(x_diff), np.array(y)


def get_data_for_test(path, fake, block):
    """
    Read the data into memory (for evaluating).
    :param path: The path of the folder containing landmarks files.
    :param fake: Assign the label of the data. Original(real) = 0, and manipulated(fake) = 1.
    :param block: The length of a 'block', i.e., the frames number of a video sample.
    :return:x: The feature vector A. It contains all the data in the datasets. Shape: [N, 136].
            x_diff; The feature vector B.  Shape: [N-1, 136]
            y: The labels. Shape: [N]
            video_y: The video-level labels (used for video-level evaluation).
            sample_to_video: A list recording the mappings of the samples(fix-length segments) to
                                their corresponding video. Shape: [N]
            count_y: A dictionary for counting the number of segments included in each video.
                                Keys: videos' name. Values: number of the segments.
    """

    files = os.listdir(path)
    x = []
    x_diff = []
    y = []

    video_y = []
    count_y = {}
    sample_to_video = []

    print("Loading data and embedding...")
    for file in tqdm(files):
        vectors = np.loadtxt(join(path, file))
        video_y.append(fake)

        for i in range(0, vectors.shape[0] - block, block):
            vec = vectors[i:i + block, :]
            x.append(vec)
            vec_next = vectors[i + 1:i + block, :]
            vec_next = np.pad(vec_next, ((0, 1), (0, 0)), 'constant', constant_values=(0, 0))
            vec_diff = (vec_next - vec)[:block - 1, :]
            x_diff.append(vec_diff)

            y.append(fake)

            # Dict for counting number of samples in video
            if file not in count_y:
                count_y[file] = 1
            else:
                count_y[file] += 1

            # Recording each samples belonging
            sample_to_video.append(file)
    return np.array(x), np.array(x_diff), np.array(y), np.array(video_y), np.array(sample_to_video), count_y


def load_data_train(add_real, add_fake, block_size, batch_size):
    train_samples, train_samples_diff, train_labels = get_data(join(add_real, "train/"), 0, block_size)
    tmp_samples, tmp_samples_diff, tmp_labels = get_data(join(add_fake, "train/"), 1, block_size)

    train_samples = torch.tensor(np.concatenate((train_samples, tmp_samples), axis=0), dtype=torch.float32)
    train_samples_diff = torch.tensor(np.concatenate((train_samples_diff, tmp_samples_diff), axis=0),
                                      dtype=torch.float32)
    train_labels = torch.tensor(np.concatenate((train_labels, tmp_labels), axis=0), dtype=torch.long)

    train_dataset_A = Data.TensorDataset(train_samples, train_labels)
    train_dataset_B = Data.TensorDataset(train_samples_diff, train_labels)

    train_iter_A = Data.DataLoader(train_dataset_A, batch_size, shuffle=True)
    train_iter_B = Data.DataLoader(train_dataset_B, batch_size, shuffle=True)

    """
    Flush the memory
    """
    tmp_samples = []
    tmp_samples_diff = []
    tmp_labels = []

    return train_iter_A, train_iter_B


def load_data_test(add_real, add_fake, block_size, batch_size):
    test_samples, test_samples_diff, test_labels, test_labels_video, test_sv, test_vc = \
        get_data_for_test(join(add_real, "test/"), 0, block_size)
    tmp_samples, tmp_samples_diff, tmp_labels, tmp_labels_video, tmp_sv, tmp_vc = \
        get_data_for_test(join(add_fake + "test/"), 1, block_size)

    test_samples = torch.tensor(np.concatenate((test_samples, tmp_samples), axis=0), dtype=torch.float32)
    test_samples_diff = torch.tensor(np.concatenate((test_samples_diff, tmp_samples_diff), axis=0), dtype=torch.float32)
    test_labels = torch.tensor(np.concatenate((test_labels, tmp_labels), axis=0), dtype=torch.long)

    test_dataset_A = Data.TensorDataset(test_samples, test_labels)
    test_dataset_B = Data.TensorDataset(test_samples_diff, test_labels)

    test_iter_A = Data.DataLoader(test_dataset_A, batch_size, shuffle=False)
    test_iter_B = Data.DataLoader(test_dataset_B, batch_size, shuffle=False)

    test_labels_video = np.concatenate((test_labels_video, tmp_labels_video), axis=0)
    test_sv = np.concatenate((test_sv, tmp_sv), axis=0)
    test_vc.update(tmp_vc)

    """
    Flush the memory
    """
    tmp_samples = []
    tmp_samples_diff = []
    tmp_labels = []

    return test_iter_A, test_iter_B, test_labels, test_labels_video, test_sv, test_vc
