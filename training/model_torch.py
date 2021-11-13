import argparse
import torch
from torch import nn
from tqdm import trange
import torch.utils.data as Data
from torch import optim
from data_utils import *
from os.path import join


class LandmarkDropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super(LandmarkDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p

    def generate_mask(self, landmark, frame):
        position_p = torch.bernoulli(torch.Tensor([1 - self.p]*(landmark//2)))
        return position_p.repeat(1, frame, 2)

    def forward(self, x: torch.Tensor):
        if self.training:
            _, frame, landmark = x.size()
            landmark_mask = self.generate_mask(landmark, frame)
            scale = 1/(1-self.p)
            return x*landmark_mask.to(x.device)*scale
        else:
            return x


class LRNet(nn.Module):
    def __init__(self, rnn_unit=64, dropout_rate=0.5):
        super(LRNet, self).__init__()

        self.dropout_landmark = LandmarkDropout(0.20)
        self.gru = nn.GRU(input_size=136, hidden_size=rnn_unit, batch_first=True, bidirectional=True)
        self.dropout_feature_1 = nn.Dropout(dropout_rate)
        self.linear_1 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.dropout_feature_2 = nn.Dropout(dropout_rate)
        self.linear_2 = nn.Linear(64, 2)
        self.output = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.dropout_landmark(x)
        x = self.gru(x)[0]
        x = x[:, -1, :]
        x = self.dropout_feature_1(x)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout_feature_2(x)
        x = self.linear_2(x)
        x = self.output(x)

        return x


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


def train(model, train_iter, test_iter, optimizer, loss, epochs, device, add_weights_file):
    log_training_loss = []
    log_training_accuracy = []
    log_testing_accuracy = []
    best_test_acc = 0.0

    model.to(device)
    model.train()
    for epoch in trange(1, epochs + 1):
        loss_sum, acc_sum, samples_sum = 0.0, 0.0, 0
        for X, y in train_iter:
            # Load data to GPU/cpu
            X = X.to(device)
            y = y.to(device)
            samples_num = X.shape[0]

            # BP
            output = model(X)
            log_softmax_output = torch.log(output)
            l = loss(log_softmax_output, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            # Generating Log
            loss_sum += l.item() * samples_num
            acc_sum += calculate_accuracy(output, y) * samples_num
            samples_sum += samples_num

        test_acc = evaluate(model, test_iter, device)
        if test_acc >= best_test_acc:
            save_hint = "\n Current test_acc:{:.4} exceeds the best record:{:.4}, " \
                        "save the model to {}".format(test_acc, best_test_acc, add_weights_file)
            torch.save(model.state_dict(), add_weights_file)
            best_test_acc = test_acc
        else:
            save_hint = "\n Current test_acc does NOT exceed the best record:{:.4}. Skip.".format(best_test_acc)

        tqdm.write("epoch:{}, loss:{:.4}, train_acc:{:.4}, test_acc:{:.4}".format(epoch, loss_sum/samples_sum,
                                                                                  acc_sum/samples_sum, test_acc)
                   + save_hint)

        log_training_loss.append(loss_sum/samples_sum)
        log_training_accuracy.append(acc_sum/samples_sum)
        log_testing_accuracy.append(test_acc)

    log = {"loss": log_training_loss, "acc_train": log_training_accuracy, "acc_test": log_testing_accuracy}
    return log


def main(args):
    if_train = args.train
    if_evaluate = args.evaluate
    if_gpu = args.gpu

    """
    Initialization
    """
    BLOCK_SIZE = 60
    BATCH_SIZE = 1024
    DROPOUT_RATE = 0.5
    RNN_UNIT = 64

    add_real = './datasets/Origin/c23/'
    add_fake = './datasets/DF/c23/'
    add_weights = './weights/torch/'

    if if_gpu:
        # Optional to uncomment if some bugs occur.
        # os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = 'cpu'

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

    if if_train:
        train_iter_A, train_iter_B = load_data_train(add_real, add_fake, BLOCK_SIZE, BATCH_SIZE)
        test_iter_A, test_iter_B, test_labels, test_labels_video, test_sv, test_vc = \
            load_data_test(add_real, add_fake, BLOCK_SIZE, BATCH_SIZE)

        # ----For g1----#
        g1 = LRNet(RNN_UNIT, DROPOUT_RATE)
        optimizer = optim.Adam(g1.parameters(), lr=0.005)
        loss = nn.NLLLoss()
        epochs = 500
        add_weights_file = join(add_weights, 'g1.pth')
        log_g1 = train(g1, train_iter_A, test_iter_A, optimizer, loss, epochs, device, add_weights_file)
        # ----g1 end----#

        # ----For g2----#
        g2 = LRNet(RNN_UNIT, DROPOUT_RATE)
        optimizer = optim.Adam(g2.parameters(), lr=0.005)
        loss = nn.NLLLoss()
        epochs = 300
        add_weights_file = join(add_weights, 'g2.pth')
        log_g2 = train(g2, train_iter_B, test_iter_B, optimizer, loss, epochs, device, add_weights_file)
        # ----g2 end----#

    if if_evaluate:
        if not if_train:
            # Test data have been loaded if trained. Thus if not trained we should load test data here.
            test_iter_A, test_iter_B, test_labels, test_labels_video, test_sv, test_vc = \
                load_data_test(add_real, add_fake, BLOCK_SIZE, BATCH_SIZE)

        g1 = LRNet(RNN_UNIT, DROPOUT_RATE)
        g2 = LRNet(RNN_UNIT, DROPOUT_RATE)

        # ----For g1----#
        g1.load_state_dict(torch.load(join(add_weights, 'g1.pth')))
        acc_g1 = evaluate(g1, test_iter_A, device)

        # ----g1 end----#

        # ----For g2----#
        g2.load_state_dict(torch.load(join(add_weights, 'g2.pth')))
        acc_g2 = evaluate(g2, test_iter_B, device)
        # ----g2 end----#

        """
        Evaluate the merged prediction (sample-level and video-level)
        """
        # ----Sample-level----#
        prediction = predict(g1, test_iter_A, device)
        prediction_diff = predict(g2, test_iter_B, device)
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
        print("Evaluation (g1) - Acc: {:.4}".format(acc_g1))
        print("Evaluation (g2) - Acc: {:.4}".format(acc_g2))
        print("Accuracy (sample-level): ", count_s / total_s)
        print("Accuracy (video-level): ", count_v / total_v)
        print("#------------End------------#")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Training and evaluating of LRNet (PyTorch version).',
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

