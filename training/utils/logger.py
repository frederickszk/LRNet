class Logger:
    def __init__(self):
        self.if_status_registered = False
        self.if_parameter_registered = False

    def register_status(self, device, dataset, add_weights):
        self.device = device
        self.dataset_name = dataset.name
        self.add_real = dataset.add_real
        self.add_fake = dataset.add_fake
        self.add_weights = add_weights
        self.if_status_registered = True

    def register_parameter(self, block_size, rnn_unit, batch_size):
        self.block_size = block_size
        self.rnn_unit = rnn_unit
        self.batch_size = batch_size
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
        print("Directory of real samples: ", self.add_real)
        print("Directory of fake samples: ", self.add_fake)
        print("Directory of model weights: ", self.add_weights)
        print("#-----Status End------#")
        print("\n")
        print("#-----Parameters------#")
        print("Block size (frames per sample): ", self.block_size)
        print("RNN hidden units: ", self.rnn_unit)
        print("Batch size: ", self.batch_size)
        print("#---Parameters End----#")
        print("\n")
        print("=======================")
        print("==      LOG END      ==")
        print("=======================")
