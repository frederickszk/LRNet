import numpy as np


def calculate_accuracy(predict, target):
    return (predict.argmax(dim=1) == target).float().mean().item()


def evaluate(model, data_iter, device):
    acc_sum, samples_sum = 0.0, 0
    model.to(device)
    model.eval()
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
