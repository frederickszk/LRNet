import numpy as np
import os
from tqdm import tqdm
from os.path import join
from utils.model import *
from configs import load_yaml


def get_data_for_test(path, fake, block):  # fake:manipulated=1 original=0
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


def predict(model, sample, device):
    model.to(device)
    model.eval()
    sample = torch.from_numpy(sample).float().to(device)
    output = model(sample)
    predictions = output.cpu().detach().numpy()
    return predictions


def merge_video_prediction(mix_prediction, s2v, vc):
    prediction_video = []
    pre_count = {}
    for p, v_label in zip(mix_prediction, s2v):
        p_bi = 0
        if p >= 0.5:
            p_bi = 1
        if v_label in pre_count:
            pre_count[v_label] += p_bi
        else:
            pre_count[v_label] = p_bi
    for key in pre_count.keys():
        prediction_video.append(pre_count[key] / vc[key])
    return prediction_video


def main():

    """
    Initialization
    """

    # Optional to uncomment if some bugs occur.
    # os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device: {}".format(device))

    args_model = load_yaml("configs/args_model.yaml")
    args_inference = load_yaml("configs/args_inference.yaml")

    BLOCK_SIZE = args_inference["BLOCK_SIZE"]
    landmark_path = args_inference["landmark_path"]

    add_weights = args_inference["add_weights"]
    weights_name_g1 = args_inference["weights_name_g1"]
    weights_name_g2 = args_inference["weights_name_g2"]

    assert os.path.exists(landmark_path), "Landmark path does not exist. Please extract the landmarks firstly."
    test_samples, test_samples_diff, _, _, test_sv, test_vc = get_data_for_test(landmark_path, 1, BLOCK_SIZE)

    g1 = LRNet(**args_model)
    g2 = LRNet(**args_model)

    print("Loading models and predicting...")

    g1.load_state_dict(torch.load(join(add_weights, weights_name_g1), map_location=device))
    g2.load_state_dict(torch.load(join(add_weights, weights_name_g2), map_location=device))

    prediction = predict(g1, test_samples, device)
    prediction_diff = predict(g2, test_samples_diff, device)

    assert len(prediction) == len(prediction_diff)
    mix_predict = []
    for i in range(len(prediction)):
        mix = prediction[i][1] + prediction_diff[i][1]
        mix_predict.append(mix/2)

    prediction_video = merge_video_prediction(mix_predict, test_sv, test_vc)

    """
    Show the results
    """
    print("\n\n", "#----Prediction Results----#")
    video_names = []
    for key in test_vc.keys():
        video_names.append(key)
    for i, pd in enumerate(prediction_video):
        if pd >= 0.5:
            label = "Fake"
        else:
            label = "Real"
        print("{}-Prediction label: {}; Scores:{}".format(video_names[i], label, pd))
    print("#------------End------------#")
    return


if __name__ == "__main__":
    main()