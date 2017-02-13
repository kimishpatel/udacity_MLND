# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
import sys
sys.path.append('../DNN/')
sys.path.append('utils/')
import matplotlib.pyplot as plt
import gc

from layercake import *
from network import *
import argparse
from data_processor import *
from generate_train_test_data import SVHN_Utils
from default_dtype import *
import configparser

num_channels = 1 # grayscale
label_index = 0
num_labels = 10
image_width = 50 #120
image_height = 50
#data_file = 'data/50x50/SVHN_data.pickle'
base_path = 'data/'
data_file = 'data/50x50/SVHN_data_shuffled.pickle'
debug_valid = 0
filter_data = False

class DataSet:
    def __init__(self, train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels):
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset

        self.train_labels = train_labels
        self.valid_labels = valid_labels
        self.test_labels = test_labels

    def reformat(self, image_width, image_height, num_channels):

        #reshape training data
        self.train_dataset = self.train_dataset.reshape(
          (-1, image_width, image_height, num_channels)).astype(DataType.ndtype)
        if label_index == 0:
            self.train_labels = (np.arange(1, num_labels+1, 1) == self.train_labels[:,None]).astype(DataType.ndtype)
        else:
            self.train_labels = ((np.append(np.arange(num_labels),-1)) == self.train_labels[:,None]).astype(DataType.ndtype)

        #reshape validation data
        self.valid_dataset = self.valid_dataset.reshape(
          (-1, image_width, image_height, num_channels)).astype(DataType.ndtype)
        if label_index == 0:
            self.valid_labels = (np.arange(1, num_labels+1, 1) == self.valid_labels[:,None]).astype(DataType.ndtype)
        else:
            self.valid_labels = ((np.append(np.arange(num_labels),-1)) == self.valid_labels[:,None]).astype(DataType.ndtype)

        #reshape test data
        self.test_dataset = self.test_dataset.reshape(
          (-1, image_width, image_height, num_channels)).astype(DataType.ndtype)
        if label_index == 0:
            self.test_labels = (np.arange(1, num_labels+1, 1) == self.test_labels[:,None]).astype(DataType.ndtype)
        else:
            self.test_labels = ((np.append(np.arange(num_labels),-1)) == self.test_labels[:,None]).astype(DataType.ndtype)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_name', type=str, help='network json file', required=True)
    parser.add_argument('-s', '--skip_training', dest = 'training', action='store_false')
    parser.add_argument('-o', '--output_file', type=str, help='output file name', required=False, default='SVHN_results.txt')
    parser.add_argument('-d', '--digit', type=str, help='digit to train on', required=False, default='0')
    parser.set_defaults(training=True)
    args = parser.parse_args()
    return args

def read_dataset(label_index, data_size=1, apply_edge_filter=False):
    print("Training on digit %d" %label_index)
    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = DataProcessor.load_and_split_data(data_file, stratify=True, label_index=label_index)

    #cut down the size of dataset if it is too bit
    if data_size < 1:
        train_dataset, train_labels = DataProcessor.chop_data(data_size, train_dataset, train_labels)
        valid_dataset, valid_labels = DataProcessor.chop_data(data_size, valid_dataset, valid_labels)
        test_dataset, test_labels = DataProcessor.chop_data(data_size, test_dataset, test_labels)

    #Find edges. Apply edge filter
    if apply_edge_filter:
        train_dataset = DataProcessor.apply_edge_filter(train_dataset)
        valid_dataset = DataProcessor.apply_edge_filter(valid_dataset)
        test_dataset = DataProcessor.apply_edge_filter(test_dataset)

    if filter_data:
        train_dataset, train_labels = DataProcessor.filter_data(train_dataset, train_labels)
        valid_dataset, valid_labels = DataProcessor.filter_data(valid_dataset, valid_labels)
        test_dataset, test_labels = DataProcessor.filter_data(test_dataset, test_labels)
        #exit(-1)

    #normalize data
    train_mean = np.mean(train_dataset)
    train_std = np.std(train_dataset)
    print("Train mean %f train std %f" %(train_mean, train_std))
    valid_mean = np.mean(valid_dataset)
    valid_std = np.std(valid_dataset)
    print("Valid mean %f train std %f" %(valid_mean, valid_std))
    test_mean = np.mean(test_dataset)
    test_std = np.std(test_dataset)
    print("Test mean %f train std %f" %(test_mean, test_std))
    #train_dataset, valid_dataset, test_dataset = DataProcessor.normalize_data([train_dataset, valid_dataset, test_dataset], train_mean, train_std)
    train_dataset, valid_dataset, test_dataset = DataProcessor.normalize_data([train_dataset, valid_dataset, test_dataset], 127.5, 255)
    #train_dataset, valid_dataset, test_dataset = DataProcessor.normalize_data([train_dataset, valid_dataset, test_dataset], train_mean, 255)
    dataset = DataSet(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels)
    return dataset

def plot_mispredicts(data, predictions, labels):
    mispredicts = (np.argmax(predictions, 1) != np.argmax(labels, 1))
    mispredict_indices = np.where(mispredicts)[0]
    fig = plt.figure()
    indices = np.random.choice(mispredict_indices, 4)
    for i in range(4):
        a = fig.add_subplot(2, 2, i+1)
        image = data[indices[i]]
        print('shape ',image.shape)
        image = DataProcessor.denormalize(image, 127.5, 255)
        image = image.squeeze()
        implot = plt.imshow(image, cmap = 'Greys_r')
        a.set_title(np.argmax(labels[indices[i]]))
    plt.show()

def load_network(args):
    print("Parsing network architecture from {}",args.file_name)
    network = Network(args.file_name)
    network.set_input_size((image_width, image_height, 1))
    #num_labels = 10 for 0 to 9 plus one if the digit does not exist
    if label_index == 0:
        network.set_output_size(num_labels)
    else:
        network.set_output_size(num_labels+1)
    cake = LayerCake(network)
    return cake, network

def train_network(args, cake, network, dataset, batch_sizes, num_iterations, accuracy_output_file, plot_mispredicts=0):

    if debug_valid:
        #SVHN_Utils.plot_images(dataset.valid_dataset, dataset.valid_labels, dataset.valid_labels.shape[0], image_width, image_height, None)
        SVHN_Utils.plot_images(dataset.test_dataset, dataset.test_labels, dataset.test_labels.shape[0], image_width, image_height, None)
        exit(-1)
    dataset.reformat(image_width, image_height, num_channels)
    gc.collect()
    print("Training set", dataset.train_dataset.shape, dataset.train_labels.shape)
    print("Validation set", dataset.valid_dataset.shape, dataset.valid_labels.shape)
    print("Test set", dataset.test_dataset.shape, dataset.test_labels.shape)

    if network.num_iterations != None:
        num_iterations = [network.num_iterations]
    if network.batch_size != None:
        batch_sizes = [network.batch_size]

    for batch_size in batch_sizes:
        for num_iter in num_iterations:
            print("batch size ", batch_size, " num iter ", num_iter)
            if args.training:
                cake.set_optimizer(network.optimizer, learning_rate = network.learning_rate, learning_rate_decay = 0.9)
                cake.max_steps_down = 5
                test_accuracy = cake.run_training(num_iter, dataset.train_dataset, dataset.train_labels, 
                                dataset.valid_dataset, dataset.valid_labels, dataset.test_dataset, dataset.test_labels, batch_size, True)
                with open(accuracy_output_file,'a') as fd:
                    fd.write("{}: {}\n".format(args.file_name, test_accuracy))
                fd.close()
            if plot_mispredicts or not args.training:
                predictions = cake.run_prediction(None, dataset.test_dataset)
                test_acc = accuracy(predictions, dataset.test_labels)
                print("Test accuracy: %.1f%%" % test_acc)

if __name__ == "__main__":
    args = parse_args()

    config = configparser.ConfigParser()
    try:
        config.read('config.ini')
        image_width = int(config['default']['image_width'])
        image_height = int(config['default']['image_height'])
    except Exception as e:
        print("could not read config file because ", str(e))
    data_file = base_path+str(image_width)+'x'+str(image_height)+'/SVHN_data_shuffled.pickle' 

    num_iterations = [20001]
    batch_sizes = [128]
    plot_mispredicts = 0
    label_index = int(args.digit)
    if label_index == 0:
        num_labels = 5
    cake, network = load_network(args)
    cake.valid_train_step = 1000
    print(args.digit)
    if network.filter_data:
        filter_data = True
    dataset = read_dataset(label_index, 1)
    output_file = args.output_file
    train_network(args, cake, network, dataset, batch_sizes, num_iterations,
                 output_file, 0)
