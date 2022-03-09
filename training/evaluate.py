import argparse
from tqdm import tqdm, trange
from torch import optim
from os.path import join
from utils.model import *
from utils.logger import Logger
from utils.metric import *
from utils.dataset import Dataset


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
    add_weights = './weights/torch/'

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
    logger.print_logs_evaluating()

    """
    Loading data
    """
    test_iter_A = None
    test_iter_B = None
    if branch_selection == 'g1':
        test_iter_A, test_labels, test_labels_video, test_sv, test_vc = \
            dataset.load_data_test_g1(BLOCK_SIZE, BATCH_SIZE)
    elif branch_selection == 'g2':
        test_iter_B, test_labels, test_labels_video, test_sv, test_vc = \
            dataset.load_data_test_g2(BLOCK_SIZE, BATCH_SIZE)
    elif branch_selection == 'all':
        test_iter_A, test_iter_B, test_labels, test_labels_video, test_sv, test_vc = \
            dataset.load_data_test_all(BLOCK_SIZE, BATCH_SIZE)
    else:
        print("Unknown branch selection:", branch_selection, '. Please check and restart')
        return

    """
    Evaluating
    """
    g1 = None
    g2 = None
    acc_g1 = None
    acc_g2 = None
    if branch_selection == 'g1' or branch_selection == 'all':
        g1 = LRNet(RNN_UNIT, DROPOUT_RATE)
        g1.load_state_dict(torch.load(join(add_weights, 'g1.pth')))
        acc_g1 = evaluate(g1, test_iter_A, device)

    if branch_selection == 'g2' or branch_selection == 'all':
        g2 = LRNet(RNN_UNIT, DROPOUT_RATE)
        g2.load_state_dict(torch.load(join(add_weights, 'g2.pth')))
        acc_g2 = evaluate(g2, test_iter_B, device)

    """
    Evaluate the [g1+g2] merged prediction (sample-level and video-level).
    If only evaluating the single branch, the sample-level evaluation 
        will be SKIPPED because it's equal to acc_g1 or acc_g2.
    Also, at this situation, the 'mix_predict' will be 
        directly equal to prediction or prediction_diff.
    """
    # ----Sample-level----#
    count_s = None
    total_s = None
    if branch_selection == 'g1':
        prediction = predict(g1, test_iter_A, device)
        mix_predict = list(prediction[:, 1])
    elif branch_selection == 'g2':
        prediction_diff = predict(g2, test_iter_B, device)
        mix_predict = list(prediction_diff[:, 1])
    else:
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

    """
    Exhibit the evaluation results.
    """
    print("\n")
    print("#----Evaluation  Results----#")
    if branch_selection == 'g1' or branch_selection == 'all':
        print("Evaluation (g1) - Acc: {:.4}".format(acc_g1))
    if branch_selection == 'g2' or branch_selection == 'all':
        print("Evaluation (g2) - Acc: {:.4}".format(acc_g2))
    if branch_selection == 'all':
        print("Accuracy (sample-level): ", count_s / total_s)
    print("Accuracy (video-level): ", count_v / total_v)
    print("#------------End------------#")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluating codes of LRNet (PyTorch version).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-g', '--gpu', action='store_true',
                        help="If use the GPU(CUDA) for evaluating."
                        )
    parser.add_argument('-d', '--dataset', type=str,
                        choices=['DF', 'F2F', 'FS', 'NT', 'FF_all'],
                        default='DF',
                        help="Select the dataset used for evaluating. "
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
                        help="Select which branch of the LRNet to be evaluated. "
                             "Valid selections: ['g1', 'g2', 'all']")
    args = parser.parse_args()
    main(args)