import json

class Network:
    def __init__(self, file):
        self.layers = []
        self.learning_rate = 0.5
        self.input_size = 784 #default size
        self.output_size = 10 #default size
        self.reg = None
        self.dropout = None
        fp = open(file)
        self.num_iterations = None
        self.filter_data = 0
        self.model_name = 'model'
        self.batch_size = None
        self.load_network(fp)

    def load_network(self, fp):
        network = json.load(fp)
        for key in network:
            if 'slice' in key:
                self.layers.append(network[key])
            if key == 'output':
                self.output = network[key]
            if key == 'learning_rate':
                self.learning_rate = network[key]
            if key == 'regularization':
                self.reg = network[key]
            if key == 'dropout':
                self.dropout = network[key]
            if key == 'optimizer':
                self.optimizer = network[key]
            if key == 'input_size':
                self.input_size = network[key]
            if key == 'output_size':
                self.output_size = network[key]
            if key == 'num_iterations':
                self.num_iterations = network[key]
            if key == 'batch_size':
                self.batch_size = network[key]
            if key == 'filter_data':
                self.filter_data = network[key]
            if key == 'model_name':
                self.model_name = network[key]

    def set_input_size(self, size):
        self.input_size = size

    def set_output_size(self, size):
        self.output_size = size
