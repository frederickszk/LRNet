import argparse
from tqdm import tqdm, trange
from torch import optim
from os.path import join
from utils.model import *
from utils.logger import Logger
from utils.data import load_data_train, load_data_test
from utils.metric import *
from utils.dataset import Dataset


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
    if_gpu = args.gpu
    dataset_name = args.dataset
    dataset_level = args.level
    branch_selection = args.branch
    """
    Initialization
    """
    BLOCK_SIZE = 60
    BATCH_SIZE = 1024
    DROPOUT_RATE = 0.5
    RNN_UNIT = 64
    EPOCHS_g1 = 100
    EPOCHS_g2 = 50
    LEARNING_RATE=0.005
    add_weights = './weights/test/'

    if if_gpu:
        # Optional to uncomment if some bugs occur.
        # os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = 'cpu'

    dataset = Dataset(name=dataset_name, level=dataset_level)

    """
    Logs
    """
    logger = Logger()
    logger.register_status(dataset=dataset,
                           device=device,
                           add_weights=add_weights,
                           branch_selection=branch_selection)
    logger.register_parameter(block_size=BLOCK_SIZE,
                              batch_size=BATCH_SIZE,
                              rnn_unit=RNN_UNIT,
                              epochs_g1=EPOCHS_g1,
                              epochs_g2=EPOCHS_g2,
                              learning_rate=LEARNING_RATE)
    logger.print_logs()

    """
    Loading data
    """
    train_iter_A = None
    train_iter_B = None
    test_iter_A = None
    test_iter_B = None
    if branch_selection == 'g1':
        train_iter_A = dataset.load_data_train_g1(BLOCK_SIZE, BATCH_SIZE)
        test_iter_A, test_labels, test_labels_video, test_sv, test_vc = \
            dataset.load_data_test_g1(BLOCK_SIZE, BATCH_SIZE)
    elif branch_selection == 'g2':
        train_iter_B = dataset.load_data_train_g2(BLOCK_SIZE, BATCH_SIZE)
        test_iter_B, test_labels, test_labels_video, test_sv, test_vc = \
            dataset.load_data_test_g2(BLOCK_SIZE, BATCH_SIZE)
    elif branch_selection == 'all':
        train_iter_A, train_iter_B = dataset.load_data_train_all(BLOCK_SIZE, BATCH_SIZE)
        test_iter_A, test_iter_B, test_labels, test_labels_video, test_sv, test_vc = \
            dataset.load_data_test_all(BLOCK_SIZE, BATCH_SIZE)
    else:
        print("Unknown branch selection:", branch_selection, '. Please check and restart')
        return

    """
    Training
    """
    if branch_selection == 'g1' or branch_selection == 'all':
        # ----For g1----#
        assert train_iter_A, test_iter_A is not None
        g1 = LRNet(RNN_UNIT, DROPOUT_RATE)
        optimizer = optim.Adam(g1.parameters(), lr=LEARNING_RATE)
        loss = nn.NLLLoss()
        add_weights_file = join(add_weights, 'g1_tmp.pth')
        log_g1 = train(g1, train_iter_A, test_iter_A, optimizer, loss, EPOCHS_g1, device, add_weights_file)
        # ----g1 end----#

    if branch_selection == 'g2' or branch_selection == 'all':
        # ----For g2----#
        assert train_iter_B, test_iter_B is not None
        g2 = LRNet(RNN_UNIT, DROPOUT_RATE)
        optimizer = optim.Adam(g2.parameters(), lr=LEARNING_RATE)
        loss = nn.NLLLoss()
        add_weights_file = join(add_weights, 'g2_tmp.pth')
        log_g2 = train(g2, train_iter_B, test_iter_B, optimizer, loss, EPOCHS_g2, device, add_weights_file)
        # ----g2 end----#

    # if if_evaluate:
    #     if not if_train:
    #         # Test data have been loaded if trained. Thus if not trained we should load test data here.
    #         test_iter_A, test_iter_B, test_labels, test_labels_video, test_sv, test_vc = \
    #             load_data_test(add_real, add_fake, BLOCK_SIZE, BATCH_SIZE)
    #
    #     g1 = LRNet(RNN_UNIT, DROPOUT_RATE)
    #     g2 = LRNet(RNN_UNIT, DROPOUT_RATE)
    #
    #     # ----For g1----#
    #     g1.load_state_dict(torch.load(join(add_weights, 'g1.pth')))
    #     acc_g1 = evaluate(g1, test_iter_A, device)
    #
    #     # ----g1 end----#
    #
    #     # ----For g2----#
    #     g2.load_state_dict(torch.load(join(add_weights, 'g2.pth')))
    #     acc_g2 = evaluate(g2, test_iter_B, device)
    #     # ----g2 end----#
    #
    #     """
    #     Evaluate the merged prediction (sample-level and video-level)
    #     """
    #     # ----Sample-level----#
    #     prediction = predict(g1, test_iter_A, device)
    #     prediction_diff = predict(g2, test_iter_B, device)
    #     count_s = 0
    #     total_s = test_labels.shape[0]
    #     mix_predict = []
    #     for i in range(len(prediction)):
    #         mix = prediction[i][1] + prediction_diff[i][1]
    #         if mix >= 1:
    #             result = 1
    #         else:
    #             result = 0
    #         if result == test_labels[i]:
    #             count_s += 1
    #         mix_predict.append(mix / 2)
    #
    #     # ----Video-level----#
    #     prediction_video = merge_video_prediction(mix_predict, test_sv, test_vc)
    #     count_v = 0
    #     total_v = len(test_labels_video)
    #     for i, pd in enumerate(prediction_video):
    #         if pd >= 0.5:
    #             result = 1
    #         else:
    #             result = 0
    #         if result == test_labels_video[i]:
    #             count_v += 1
    #
    #     print("\n")
    #     print("#----Evaluation  Results----#")
    #     print("Evaluation (g1) - Acc: {:.4}".format(acc_g1))
    #     print("Evaluation (g2) - Acc: {:.4}".format(acc_g2))
    #     print("Accuracy (sample-level): ", count_s / total_s)
    #     print("Accuracy (video-level): ", count_v / total_v)
    #     print("#------------End------------#")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Training and evaluating of LRNet (PyTorch version).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-g', '--gpu', action='store_true',
                        help="If use the GPU(CUDA) for training."
                        )
    parser.add_argument('-d', '--dataset', type=str,
                        choices=['DF', 'F2F', 'FS', 'NT', 'FF_all'],
                        default='DF',
                        help="Select the dataset used for training. "
                             "Valid selections: ['DF', 'F2F', 'FS', 'NT', 'FF_all'] "
                        )
    parser.add_argument('-l', '--level', type=str,
                        choices=['raw', 'c23', 'c40'],
                        default='c23',
                        help="Select the dataset compression level. "
                             "Valid selections: ['raw', 'c23', 'c40'] ")
    parser.add_argument('-b', '--branch', type=str,
                        choices=['g1', 'g2', 'all'],
                        default='all',
                        help="Select which branch of the LRNet to be trained. "
                             "Valid selections: ['g1', 'g2', 'all']")
    args = parser.parse_args()
    main(args)