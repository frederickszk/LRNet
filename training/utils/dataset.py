from os.path import join, exists
import numpy as np
import torch
import torch.utils.data as Data
from .data import get_data, get_data_for_test


class Dataset:
    def __init__(self, name, level):
        self.name = name
        self.level = level
        # self.block_size = block_size
        # self.batch_size = batch_size
        self.add_root = './datasets'  # Modify this if you change the root address of the dataset.
        self.add_real = []
        self.add_fake = []
        self.if_inited = False
        self.init_add()
        assert self.if_inited

    def init_add(self):
        if self.name in ['DF', 'NT', 'F2F', 'FS']:
            self.add_real.append(join(self.add_root, 'Origin', self.level))
            self.add_fake.append(join(self.add_root, self.name, self.level))
        elif self.name == 'FF_all':
            for name_sub_dataset in ['DF', 'NT', 'F2F', 'FS']:
                self.add_real.append(join(self.add_root, 'Origin', self.level))
                self.add_fake.append(join(self.add_root, name_sub_dataset, self.level))
        else:
            print("Unsupported dataset name:", self.name, ". Please check and restart.")
            return

        # Ensure the dataset directory exists.
        for add in self.add_real:
            if not exists(add):
                print("The dataset directory:", add, "does not exist. Please check and restart.")
                return
        for add in self.add_fake:
            if not exists(add):
                print("The dataset directory:", add, "does not exist. Please check and restart.")
                return

        self.if_inited = True

    def load_data_train_all(self, block_size, batch_size):
        train_samples = None
        train_samples_diff = None
        train_labels = None

        for add_r in self.add_real:
            real_samples, real_samples_diff, real_labels = get_data(join(add_r, "train/"), 0, block_size)
            if train_samples is None:
                train_samples = real_samples
                train_samples_diff = real_samples_diff
                train_labels = real_labels
            else:
                train_samples = np.concatenate((train_samples, real_samples), axis=0)
                train_samples_diff = np.concatenate((train_samples_diff, real_samples_diff), axis=0)
                train_labels = np.concatenate((train_labels, real_labels), axis=0)

        # Flush the memory
        real_samples = None
        real_samples_diff = None
        real_labels = None

        for add_f in self.add_fake:
            fake_samples, fake_samples_diff, fake_labels = get_data(join(add_f, "train/"), 1, block_size)
            train_samples = np.concatenate((train_samples, fake_samples), axis=0)
            train_samples_diff = np.concatenate((train_samples_diff, fake_samples_diff), axis=0)
            train_labels = np.concatenate((train_labels, fake_labels), axis=0)

        # Flush the memory
        fake_samples = None
        fake_samples_diff = None
        fake_labels = None

        # Convert to PyTorch dataset
        train_samples = torch.tensor(train_samples, dtype=torch.float32)
        train_samples_diff = torch.tensor(train_samples_diff, dtype=torch.float32)
        train_labels = torch.tensor(train_labels, dtype=torch.long)

        train_dataset_A = Data.TensorDataset(train_samples, train_labels)
        train_dataset_B = Data.TensorDataset(train_samples_diff, train_labels)
        train_iter_A = Data.DataLoader(train_dataset_A, batch_size, shuffle=True)
        train_iter_B = Data.DataLoader(train_dataset_B, batch_size, shuffle=True)

        return train_iter_A, train_iter_B

    def load_data_train_g1(self, block_size, batch_size):
        train_samples = None
        train_labels = None

        for add_r in self.add_real:
            real_samples, _, real_labels = get_data(join(add_r, "train/"), 0, block_size)
            if train_samples is None:
                train_samples = real_samples
                train_labels = real_labels
            else:
                train_samples = np.concatenate((train_samples, real_samples), axis=0)
                train_labels = np.concatenate((train_labels, real_labels), axis=0)

        # Flush the memory
        real_samples = None
        real_labels = None

        for add_f in self.add_fake:
            fake_samples, _, fake_labels = get_data(join(add_f, "train/"), 1, block_size)
            train_samples = np.concatenate((train_samples, fake_samples), axis=0)
            train_labels = np.concatenate((train_labels, fake_labels), axis=0)

        # Flush the memory
        fake_samples = None
        fake_labels = None

        # Convert to PyTorch dataset
        train_samples = torch.tensor(train_samples, dtype=torch.float32)
        train_labels = torch.tensor(train_labels, dtype=torch.long)

        train_dataset_A = Data.TensorDataset(train_samples, train_labels)
        train_iter_A = Data.DataLoader(train_dataset_A, batch_size, shuffle=True)

        return train_iter_A

    def load_data_train_g2(self, block_size, batch_size):
        train_samples_diff = None
        train_labels = None

        for add_r in self.add_real:
            _, real_samples_diff, real_labels = get_data(join(add_r, "train/"), 0, block_size)
            if train_samples_diff is None:
                train_samples_diff = real_samples_diff
                train_labels = real_labels
            else:
                train_samples_diff = np.concatenate((train_samples_diff, real_samples_diff), axis=0)
                train_labels = np.concatenate((train_labels, real_labels), axis=0)

        # Flush the memory
        real_samples_diff = None
        real_labels = None

        for add_f in self.add_fake:
            _, fake_samples_diff, fake_labels = get_data(join(add_f, "train/"), 1, block_size)
            train_samples_diff = np.concatenate((train_samples_diff, fake_samples_diff), axis=0)
            train_labels = np.concatenate((train_labels, fake_labels), axis=0)

        # Flush the memory
        fake_samples_diff = None
        fake_labels = None

        # Convert to PyTorch dataset
        train_samples_diff = torch.tensor(train_samples_diff, dtype=torch.float32)
        train_labels = torch.tensor(train_labels, dtype=torch.long)

        train_dataset_B = Data.TensorDataset(train_samples_diff, train_labels)
        train_iter_B = Data.DataLoader(train_dataset_B, batch_size, shuffle=True)

        return train_iter_B

    def load_data_test_all(self, block_size, batch_size):
        test_samples = None
        test_samples_diff = None
        test_labels = None
        test_labels_video = None
        test_sv = None
        test_vc = None

        for add_r in self.add_real:
            real_samples, real_samples_diff, real_labels, real_labels_video, real_sv, real_vc = \
                get_data_for_test(join(add_r, "test/"), 0, block_size)
            if test_samples is None:
                test_samples = real_samples
                test_samples_diff = real_samples_diff
                test_labels = real_labels
                test_labels_video = real_labels_video
                test_sv = real_sv
                test_vc = real_vc
            else:
                test_samples = np.concatenate((test_samples, real_samples), axis=0)
                test_samples_diff = np.concatenate((test_samples_diff, real_samples_diff), axis=0)
                test_labels = np.concatenate((test_labels, real_labels), axis=0)
                test_labels_video = np.concatenate((test_labels_video, real_labels_video), axis=0)
                test_sv = np.concatenate((test_sv, real_sv), axis=0)
                test_vc.update(real_sv)

        # Flush the memory
        real_samples = None
        real_samples_diff = None
        real_labels = None

        for add_f in self.add_fake:
            fake_samples, fake_samples_diff, fake_labels, fake_labels_video, fake_sv, fake_vc = \
                get_data_for_test(join(add_f, "test/"), 1, block_size)
            test_samples = np.concatenate((test_samples, fake_samples), axis=0)
            test_samples_diff = np.concatenate((test_samples_diff, fake_samples_diff), axis=0)
            test_labels = np.concatenate((test_labels, fake_labels), axis=0)
            test_labels_video = np.concatenate((test_labels_video, fake_labels_video), axis=0)
            test_sv = np.concatenate((test_sv, fake_sv), axis=0)
            test_vc.update(fake_vc)

        # Flush the memory
        fake_samples = None
        fake_samples_diff = None
        fake_labels = None

        # Convert to PyTorch dataset
        test_samples = torch.tensor(test_samples, dtype=torch.float32)
        test_samples_diff = torch.tensor(test_samples_diff, dtype=torch.float32)
        test_labels = torch.tensor(test_labels, dtype=torch.long)

        test_dataset_A = Data.TensorDataset(test_samples, test_labels)
        test_dataset_B = Data.TensorDataset(test_samples_diff, test_labels)

        test_iter_A = Data.DataLoader(test_dataset_A, batch_size, shuffle=True)
        test_iter_B = Data.DataLoader(test_dataset_B, batch_size, shuffle=True)

        return test_iter_A, test_iter_B, test_labels, test_labels_video, test_sv, test_vc

    def load_data_test_g1(self, block_size, batch_size):
        test_samples = None
        test_labels = None
        test_labels_video = None
        test_sv = None
        test_vc = None

        for add_r in self.add_real:
            real_samples, _, real_labels, real_labels_video, real_sv, real_vc = \
                get_data_for_test(join(add_r, "test/"), 0, block_size)
            if test_samples is None:
                test_samples = real_samples
                test_labels = real_labels
                test_labels_video = real_labels_video
                test_sv = real_sv
                test_vc = real_vc
            else:
                test_samples = np.concatenate((test_samples, real_samples), axis=0)
                test_labels = np.concatenate((test_labels, real_labels), axis=0)
                test_labels_video = np.concatenate((test_labels_video, real_labels_video), axis=0)
                test_sv = np.concatenate((test_sv, real_sv), axis=0)
                test_vc.update(real_sv)

        # Flush the memory
        real_samples = None
        real_labels = None

        for add_f in self.add_fake:
            fake_samples, _, fake_labels, fake_labels_video, fake_sv, fake_vc = \
                get_data_for_test(join(add_f, "test/"), 1, block_size)
            test_samples = np.concatenate((test_samples, fake_samples), axis=0)
            test_labels = np.concatenate((test_labels, fake_labels), axis=0)
            test_labels_video = np.concatenate((test_labels_video, fake_labels_video), axis=0)
            test_sv = np.concatenate((test_sv, fake_sv), axis=0)
            test_vc.update(fake_vc)

        # Flush the memory
        fake_samples = None
        fake_labels = None

        # Convert to PyTorch dataset
        test_samples = torch.tensor(test_samples, dtype=torch.float32)
        test_labels = torch.tensor(test_labels, dtype=torch.long)

        test_dataset_A = Data.TensorDataset(test_samples, test_labels)
        test_iter_A = Data.DataLoader(test_dataset_A, batch_size, shuffle=True)

        return test_iter_A, test_labels, test_labels_video, test_sv, test_vc

    def load_data_test_g2(self, block_size, batch_size):
        test_samples_diff = None
        test_labels = None
        test_labels_video = None
        test_sv = None
        test_vc = None

        for add_r in self.add_real:
            _, real_samples_diff, real_labels, real_labels_video, real_sv, real_vc = \
                get_data_for_test(join(add_r, "test/"), 0, block_size)
            if test_samples_diff is None:
                test_samples_diff = real_samples_diff
                test_labels = real_labels
                test_labels_video = real_labels_video
                test_sv = real_sv
                test_vc = real_vc
            else:
                test_samples_diff = np.concatenate((test_samples_diff, real_samples_diff), axis=0)
                test_labels = np.concatenate((test_labels, real_labels), axis=0)
                test_labels_video = np.concatenate((test_labels_video, real_labels_video), axis=0)
                test_sv = np.concatenate((test_sv, real_sv), axis=0)
                test_vc.update(real_sv)

        # Flush the memory
        real_samples_diff = None
        real_labels = None

        for add_f in self.add_fake:
            _, fake_samples_diff, fake_labels, fake_labels_video, fake_sv, fake_vc = \
                get_data_for_test(join(add_f, "test/"), 1, block_size)
            test_samples_diff = np.concatenate((test_samples_diff, fake_samples_diff), axis=0)
            test_labels = np.concatenate((test_labels, fake_labels), axis=0)
            test_labels_video = np.concatenate((test_labels_video, fake_labels_video), axis=0)
            test_sv = np.concatenate((test_sv, fake_sv), axis=0)
            test_vc.update(fake_vc)

        # Flush the memory
        fake_samples_diff = None
        fake_labels = None

        # Convert to PyTorch dataset
        test_samples_diff = torch.tensor(test_samples_diff, dtype=torch.float32)
        test_labels = torch.tensor(test_labels, dtype=torch.long)

        test_dataset_B = Data.TensorDataset(test_samples_diff, test_labels)
        test_iter_B = Data.DataLoader(test_dataset_B, batch_size, shuffle=True)

        return test_iter_B, test_labels, test_labels_video, test_sv, test_vc
