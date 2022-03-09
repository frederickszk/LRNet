class Logger:
    def __init__(self):
        self.if_status_registered = False
        self.if_parameter_registered = False

        # Declare the variables to be registered
        self.device = None
        self.dataset_name = None
        self.dataset_level = None
        self.add_real = None
        self.add_fake = None
        self.add_weights = None
        self.branch_selection = None

        self.block_size = None
        self.batch_size = None
        self.rnn_unit = None
        self.epochs_g1 = None
        self.epochs_g2 = None
        self.learning_rate = None

    def register_status(self, dataset, **kwargs):
        self.dataset_name = dataset.name
        self.dataset_level = dataset.level
        self.add_real = dataset.add_real
        self.add_fake = dataset.add_fake
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

        self.if_status_registered = True

    def register_parameter(self, **kwargs):
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])
        self.if_parameter_registered = True

    def print_logs(self):
        if not self.if_status_registered:
            print("Please register the status.")
            return

        if not self.if_parameter_registered:
            print("Please register the parameters.")
            return

        print("=======================")
        print("==        LOG        ==")
        print("=======================")
        print("\n")
        print("#-------Status--------#")
        print("Using device: ", self.device)
        print("Dataset name: ", self.dataset_name)
        print("Dataset compression level: ", self.dataset_level)
        print("Directory of real samples: ", self.add_real)
        print("Directory of fake samples: ", self.add_fake)
        print("Directory of model weights: ", self.add_weights)
        print("Which branch of the LRNet to be trained: ", self.branch_selection)
        print("#-----Status End------#")
        print("\n")
        print("#-----Parameters------#")
        print("Block size (frames per sample): ", self.block_size)
        print("Batch size: ", self.batch_size)
        print("RNN hidden units: ", self.rnn_unit)
        print("Training epochs (g1): ", self.epochs_g1)
        print("Training epochs (g2): ", self.epochs_g2)
        print("Learning rate ", self.learning_rate)
        print("#---Parameters End----#")
        print("\n")
        print("=======================")
        print("==      LOG END      ==")
        print("=======================")
