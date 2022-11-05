import numpy as np
import matplotlib.pyplot as plt
import torch


def calculate_accuracy(predict, target):
    return (predict.argmax(dim=1) == target).float().mean().item()


def evaluate(model, data_iter, device):
    acc_sum, samples_sum = 0.0, 0
    model.to(device)
    model.eval()
    with torch.no_grad():
        for X, y in data_iter:
            X = X.to(device)
            y = y.to(device)
            samples_num = X.shape[0]
            acc_sum += calculate_accuracy(model(X), y) * samples_num
            samples_sum += samples_num
    model.train()
    return acc_sum/samples_sum


def predict(model, data_iter, device):
    predictions = []
    model.to(device)
    model.eval()
    for X, _ in data_iter:
        X = X.to(device)
        output = model(X)
        prediction_batch = output.cpu().detach().numpy()
        predictions.append(prediction_batch)
    model.train()
    prediction_all = np.concatenate(predictions, axis=0)
    return prediction_all


def merge_video_prediction(mix_prediction, s2v, vc):
    """
    :param mix_prediction: The mixed prediction of 2 branches. (of each sample)
    :param s2v: Sample-to-video. Refer to the 'sample_to_video' in function get_data_for_test()
    :param vc: Video-Count. Refer to the 'count_y' in function get_data_for_test()
    :return: prediction_video: The prediction of each video.
    """
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


def plot_ROC(fpr, tpr, roc_auc):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label="ROC curve (area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    plt.show()
