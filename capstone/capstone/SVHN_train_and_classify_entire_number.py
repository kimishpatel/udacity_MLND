# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
import sys
sys.path.append('../DNN/')
import matplotlib.pyplot as plt
import gc

from layercake_SVHN import *
from network import *
import argparse
from data_processor import *
from default_dtype import *
import configparser

image_width = 50
image_height = 50
num_labels = 10
num_channels = 1 # grayscale
base_path = 'data/'
data_file = 'data/50x50/SVHN_data_shuffled.pickle'
num_digits = 5

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
        labels_temp = np.ndarray(shape=(self.train_labels.shape[0], (num_digits+1)*11))
        labels_temp[:,0:num_labels+1] = ((np.append(np.arange(num_labels),-1)) == self.train_labels[:,0].reshape(-1,1)).astype(DataType.ndtype)
        start_index = num_labels+1
        end_index = start_index+num_labels+1
        for i in range(0, num_digits, 1):
            temp = ((np.append(np.arange(num_labels),-1)) == self.train_labels[:,i+1].reshape(-1,1)).astype(DataType.ndtype)
            labels_temp[:,start_index:end_index] = temp
            start_index = end_index
            end_index = start_index+num_labels+1
        self.train_labels = labels_temp

        #reshape validation data
        self.valid_dataset = self.valid_dataset.reshape(
          (-1, image_width, image_height, num_channels)).astype(DataType.ndtype)
        labels_temp = np.ndarray(shape=(self.valid_labels.shape[0], (num_digits+1)*11))
        labels_temp[:,0:num_labels+1] = ((np.append(np.arange(num_labels),-1)) == self.valid_labels[:,0].reshape(-1,1)).astype(DataType.ndtype)
        start_index = num_labels+1
        end_index = start_index+num_labels+1
        for i in range(0, num_digits, 1):
            temp = ((np.append(np.arange(num_labels),-1)) == self.valid_labels[:,i+1].reshape(-1,1)).astype(DataType.ndtype)
            labels_temp[:,start_index:end_index] = temp
            start_index = end_index
            end_index = start_index+num_labels+1
        self.valid_labels = labels_temp

        #reshape test data
        self.test_dataset = self.test_dataset.reshape(
          (-1, image_width, image_height, num_channels)).astype(DataType.ndtype)
        labels_temp = np.ndarray(shape=(self.test_labels.shape[0], (num_digits+1)*11))
        labels_temp[:,0:num_labels+1] = ((np.append(np.arange(num_labels),-1)) == self.test_labels[:,0].reshape(-1,1)).astype(DataType.ndtype)
        start_index = num_labels+1
        end_index = start_index+num_labels+1
        for i in range(0, num_digits, 1):
            temp = ((np.append(np.arange(num_labels),-1)) == self.test_labels[:,i+1].reshape(-1,1)).astype(DataType.ndtype)
            labels_temp[:,start_index:end_index] = temp
            start_index = end_index
            end_index = start_index+num_labels+1
        self.test_labels = labels_temp

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_name', type=str, help='network json file', required=True)
    parser.add_argument('-s', '--skip_training', dest = 'training', action='store_false')
    parser.set_defaults(training=True)
    args = parser.parse_args()
    return args

def read_dataset(data_size=1, apply_edge_filter=False):
    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = DataProcessor.load_and_split_data(data_file, stratify=False)

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

    #normalize data
    train_dataset, valid_dataset, test_dataset = DataProcessor.normalize_data([train_dataset, valid_dataset, test_dataset], 127.5, 255)
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
    network.set_output_size((num_digits+1, num_labels+1))
    cake = LayerCake(network)
    return cake, network

def train_network(args, cake, network, dataset, batch_sizes, num_iterations, accuracy_output_file, plot_mispredicts=0):

    dataset.reformat(image_width, image_height, num_channels)
    gc.collect()
    print('Training set', dataset.train_dataset.shape, dataset.train_labels.shape)
    print('Validation set', dataset.valid_dataset.shape, dataset.valid_labels.shape)
    print('Test set', dataset.test_dataset.shape, dataset.test_labels.shape)

    if network.num_iterations != None:
        num_iterations = [network.num_iterations]
    if network.batch_size != None:
        batch_sizes = [network.batch_size]

    for batch_size in batch_sizes:
        for num_iter in num_iterations:
            print("batch size ", batch_size, " num iter ", num_iter)
            if args.training:
                cake.set_optimizer(network.optimizer, learning_rate = network.learning_rate, learning_rate_decay = 0.96)
                cake.max_steps_down = 30
                test_accuracy = cake.run_training(num_iter, dataset.train_dataset, dataset.train_labels, 
                                dataset.valid_dataset, dataset.valid_labels, dataset.test_dataset, dataset.test_labels, batch_size, save_model=True)
		with open(accuracy_output_file,'a') as fd:
		    fd.write("{}: {}\n".format(args.file_name, test_accuracy))
		fd.close()
            if plot_mispredicts:
                predictions = cake.run_prediction(None, dataset.test_dataset)
                #predictions = cake.run_prediction(cake.session, test_dataset)
                test_acc = accuracy(predictions, dataset.test_labels)
                print("Test accuracy: %.1f%%" % test_acc)
                plot_mispredicts(dataset.test_dataset, predictions, dataset.test_labels)

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
    cake, network = load_network(args)
    cake.valid_train_step = 1000
    dataset = read_dataset(1)
    train_network(args, cake, network, dataset, batch_sizes, num_iterations,
                 'SVHN_results_entire_number.txt')
