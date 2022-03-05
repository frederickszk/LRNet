from os.path import join, exists


class Dataset:
    def __init__(self, name, level, block_size, batch_size):
        self.name = name
        self.level = level
        self.block_size = block_size
        self.batch_size = batch_size
        self.add_root = './datasets'  # Modify this if you change the root address of the dataset.
        self.add_real = []
        self.add_fake = []
        self.if_inited = False
        self.init_add()
        assert self.if_inited

    def init_add(self):
        if self.name in ['DF', 'FS', 'F2F', 'FS']:
            self.add_real.append(join(self.add_root, 'Origin', self.level))
            self.add_fake.append(join(self.add_root, self.name, self.level))
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
