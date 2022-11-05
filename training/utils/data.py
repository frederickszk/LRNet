import os
from os.path import join
import numpy as np
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

    print("Loading data from: ", path)
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
    # :param dataset_name: The name of the dataset that files belong to.
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

    print("Loading data from: ", path)
    for file in tqdm(files):
        vectors = np.loadtxt(join(path, file))

        """
        [22/08/21] Add a judgement here, when the length of the vectors is smaller than the block, just discard it.]
        """
        if vectors.shape[0] < block:
            continue
        for i in range(0, vectors.shape[0] - block, block):
            vec = vectors[i:i + block, :]
            x.append(vec)
            vec_next = vectors[i + 1:i + block, :]
            vec_next = np.pad(vec_next, ((0, 1), (0, 0)), 'constant', constant_values=(0, 0))
            vec_diff = (vec_next - vec)[:block - 1, :]
            x_diff.append(vec_diff)

            y.append(fake)

            # Dict for counting number of samples in video
            file_dir = join(path, file)
            if file_dir not in count_y:
                count_y[file_dir] = 1
            else:
                count_y[file_dir] += 1

            # Recording each samples belonging
            sample_to_video.append(file_dir)

        video_y.append(fake)
    return np.array(x), np.array(x_diff), np.array(y), np.array(video_y), np.array(sample_to_video), count_y