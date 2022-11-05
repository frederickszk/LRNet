class Logger:
    def __init__(self):
        self.if_status_registered = False
        self.if_args_registered = False

        # Declare the variables to be registered
        """
        [Status]
        Determined by the user's cmd-args when run the scripts
        """
        self.device = None
        self.dataset_add_root = None
        self.dataset_name = None
        self.dataset_level = None
        self.add_real = None
        self.add_fake = None
        self.branch_selection = None

        """
        [Train Arguments]
        Customized by './configs/args_train.yaml'
        Tips: dataset_add_root actually defined in this config file.
        """
        self.BLOCK_SIZE = None
        self.BATCH_SIZE = None
        self.add_weights = None
        self.EPOCHS_g1 = None
        self.LEARNING_RATE_g1 = None
        self.weights_name_g1 = None
        self.EPOCHS_g2 = None
        self.LEARNING_RATE_g2 = None
        self.weights_name_g2 = None

        """
        [Model Arguments]
        Customized by './configs/args_model.yaml'
        """
        self.lm_dropout_rate = None
        self.rnn_unit = None
        self.fc_dropout_rate = None
        self.res_hidden = None

    def register_status(self, dataset, **kwargs):
        self.dataset_add_root = dataset.add_root
        self.dataset_name = dataset.name
        self.dataset_level = dataset.level
        self.add_real = dataset.add_real
        self.add_fake = dataset.add_fake
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

        self.if_status_registered = True

    def register_args(self, **kwargs):
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])
        self.if_args_registered = True

    def print_logs_training(self):
        if not self.if_status_registered:
            print("Please register the status.")
            return

        if not self.if_args_registered:
            print("Please register the arguments.")
            return

        print("=========================")
        print("==         LOG         ==")
        print("=========================")
        print()
        print("#--------Status---------#")
        print("Using device: ", self.device)
        print("Dataset name: ", self.dataset_name)
        print("Dataset compression level: ", self.dataset_level)
        print("Dataset root directory: ", self.dataset_add_root)
        print("Directory of real samples: ", self.add_real)
        print("Directory of fake samples: ", self.add_fake)
        print("Which branch of the LRNet to be trained: ", self.branch_selection)
        print("#------Status End-------#")
        print()
        print("#----Train Arguments----#")
        print("Block size (frames per sample): ", self.BLOCK_SIZE)
        print("Batch size: ", self.BATCH_SIZE)
        print("Directory of model weights: ", self.add_weights)
        print("Training epochs (g1): ", self.EPOCHS_g1)
        print("Learning rate (g1): ", self.LEARNING_RATE_g1)
        print("Saved weights name (g1): ", self.weights_name_g1)
        print("Training epochs (g2): ", self.EPOCHS_g2)
        print("Learning rate (g2): ", self.LEARNING_RATE_g2)
        print("Saved weights name (g2): ", self.weights_name_g2)
        print("#--Train Arguments End--#")
        print()
        print("#----Model Arguments----#")
        print("RNN hidden units: ", self.rnn_unit)
        print("FC-layer hidden units: ", self.res_hidden)
        print("Landmark dropout rate: ", self.lm_dropout_rate)
        print("FC-layer dropout rate: ", self.fc_dropout_rate)
        print("#--Model Arguments End--#")
        print()
        print("=========================")
        print("==       LOG END       ==")
        print("=========================")

    def print_logs_evaluating(self):
        if not self.if_status_registered:
            print("Please register the status.")
            return
        if not self.if_args_registered:
            print("Please register the arguments.")
            return

        print("=========================")
        print("==         LOG         ==")
        print("=========================")
        print()
        print("#--------Status---------#")
        print("Using device: ", self.device)
        print("Dataset name: ", self.dataset_name)
        print("Dataset compression level: ", self.dataset_level)
        print("Dataset root directory: ", self.dataset_add_root)
        print("Directory of real samples: ", self.add_real)
        print("Directory of fake samples: ", self.add_fake)
        print("Which branch of the LRNet to be evaluated: ", self.branch_selection)
        print("#------Status End-------#")
        print()
        print("#---Evaluate Arguments---#")
        print("Block size (frames per sample): ", self.BLOCK_SIZE)
        print("Batch size: ", self.BATCH_SIZE)
        print("Directory of model weights: ", self.add_weights)
        print("Saved weights name (g1): ", self.weights_name_g1)
        print("Saved weights name (g2): ", self.weights_name_g2)
        print("#-Evaluate Arguments End-#")
        print()
        print("#----Model Arguments----#")
        print("RNN hidden units: ", self.rnn_unit)
        print("FC-layer hidden units: ", self.res_hidden)
        print("Landmark dropout rate: ", self.lm_dropout_rate)
        print("FC-layer dropout rate: ", self.fc_dropout_rate)
        print("#--Model Arguments End--#")
        print()
        print("=========================")
        print("==       LOG END       ==")
        print("=========================")
