import argparse
import os
from tqdm import tqdm, trange
from torch import optim
from os.path import join
from utils.model import *
from utils.logger import Logger
from utils.metric import *
from utils.dataset import Dataset
from configs.loader import load_yaml


def train_loop(model, train_iter, val_iter, optimizer, loss, epochs, device, add_weights_file):
    log_training_loss = []
    log_training_accuracy = []
    log_val_accuracy = []
    best_val_acc = 0.0

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
        train_acc = acc_sum/samples_sum

        val_acc = evaluate(model, val_iter, device)

        if val_acc >= best_val_acc:
            save_hint = "save the model to {}".format(add_weights_file)
            torch.save(model.state_dict(), add_weights_file)
            best_val_acc = val_acc
        else:
            save_hint = ""

        tqdm.write("epoch:{}, loss:{:.4}, train_acc:{:.4}, test_acc:{:.4}, best_record:{:.4}  "
                   .format(epoch, loss_sum / samples_sum, train_acc, val_acc, best_val_acc)
                   + save_hint)

        log_training_loss.append(loss_sum/samples_sum)
        log_training_accuracy.append(train_acc)
        log_val_accuracy.append(val_acc)

    log = {"loss": log_training_loss, "acc_train": log_training_accuracy, "acc_val": log_val_accuracy}
    return log


def main(args):
    if_gpu = args.gpu
    dataset_name = args.dataset
    dataset_level = args.level
    branch_selection = args.branch
    """
    Initialization
    """
    args_model = load_yaml("configs/args_model.yaml")
    args_train = load_yaml("configs/args_train.yaml")

    BLOCK_SIZE = args_train["BLOCK_SIZE"]
    BATCH_SIZE = args_train["BATCH_SIZE"]

    add_weights = args_train["add_weights"]
    if not os.path.exists(add_weights):
        os.makedirs(add_weights)

    EPOCHS_g1 = args_train["EPOCHS_g1"]
    LEARNING_RATE_g1 = args_train["LEARNING_RATE_g1"]
    weights_name_g1 = args_train["weights_name_g1"]

    EPOCHS_g2 = args_train["EPOCHS_g2"]
    LEARNING_RATE_g2 = args_train["LEARNING_RATE_g2"]
    weights_name_g2 = args_train["weights_name_g2"]

    if if_gpu:
        # Optional to uncomment if some bugs occur.
        # os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = 'cpu'

    dataset = Dataset(add_root=args_train["add_dataset_root"],
                      name=dataset_name, level=dataset_level)

    """
    Logs
    """
    logger = Logger()
    logger.register_status(dataset=dataset,
                           device=device,
                           branch_selection=branch_selection)
    logger.register_args(**args_train, **args_model)
    logger.print_logs_training()

    """
    Loading data
    """
    train_iter_A = None
    train_iter_B = None
    val_iter_A = None
    val_iter_B = None
    if branch_selection == 'g1':
        train_iter_A = dataset.load_data_train_g1(BLOCK_SIZE, BATCH_SIZE)
        val_iter_A = dataset.load_data_val_g1(BLOCK_SIZE, BATCH_SIZE)
    elif branch_selection == 'g2':
        train_iter_B = dataset.load_data_train_g2(BLOCK_SIZE, BATCH_SIZE)
        val_iter_B = dataset.load_data_val_g2(BLOCK_SIZE, BATCH_SIZE)
    elif branch_selection == 'all':
        train_iter_A, train_iter_B = dataset.load_data_train_all(BLOCK_SIZE, BATCH_SIZE)
        val_iter_A, val_iter_B = dataset.load_data_val_all(BLOCK_SIZE, BATCH_SIZE)
    else:
        print("Unknown branch selection:", branch_selection, '. Please check and restart')
        return

    """
    Training
    """
    if branch_selection == 'g1' or branch_selection == 'all':
        assert train_iter_A, val_iter_A is not None
        g1 = LRNet(**args_model)
        optimizer = optim.Adam(g1.parameters(), lr=LEARNING_RATE_g1)
        loss = nn.NLLLoss()
        add_weights_file = join(add_weights, weights_name_g1)
        log_g1 = train_loop(g1, train_iter_A, val_iter_A, optimizer, loss, EPOCHS_g1, device, add_weights_file)

    if branch_selection == 'g2' or branch_selection == 'all':
        assert train_iter_B, val_iter_B is not None
        g2 = LRNet(**args_model)
        optimizer = optim.Adam(g2.parameters(), lr=LEARNING_RATE_g2)
        loss = nn.NLLLoss()
        add_weights_file = join(add_weights, weights_name_g2)
        log_g2 = train_loop(g2, train_iter_B, val_iter_B, optimizer, loss, EPOCHS_g2, device, add_weights_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Training codes of LRNet (PyTorch version).',
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
                        default='raw',
                        help="Select the dataset compression level. "
                             "Valid selections: ['raw', 'c23', 'c40'] ")
    parser.add_argument('-b', '--branch', type=str,
                        choices=['g1', 'g2', 'all'],
                        default='all',
                        help="Select which branch of the LRNet to be trained. "
                             "Valid selections: ['g1', 'g2', 'all']")
    args = parser.parse_args()
    main(args)