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
import random
import configparser

num_channels = 1 # grayscale
label_index = 0
num_labels = 10
image_width = 50 #120
image_height = 50
base_path = 'data/'
data_file = 'data/50x50/SVHN_data_shuffled.pickle'

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
        if self.train_dataset != None:
            self.train_dataset = self.train_dataset.reshape(
              (-1, image_width, image_height, num_channels)).astype(DataType.ndtype)

        #reshape validation data
        if self.valid_dataset != None:
            self.valid_dataset = self.valid_dataset.reshape(
              (-1, image_width, image_height, num_channels)).astype(DataType.ndtype)

        #reshape test data
        self.test_dataset = self.test_dataset.reshape(
          (-1, image_width, image_height, num_channels)).astype(DataType.ndtype)

def read_dataset(read_extra=False):
    print("Testing on digit %d" %label_index)
    if read_extra:
        _, _, test_dataset, test_labels = DataProcessor.load_data(data_file, True)
    else:
        train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = DataProcessor.load_and_split_data(data_file, stratify=False, label_index=None)

    [test_dataset] = DataProcessor.normalize_data([test_dataset], 127.5, 255)
    dataset = DataSet(None, None, None, None, test_dataset, test_labels)
    return dataset

def plot_mispredicts(data, predictions, mispredicts, labels, mispredict=1):
    if mispredict:
        mispredict_indices = np.where(mispredicts != 1)[0]
    else:
        mispredict_indices = np.where(mispredicts == 1)[0]
    fig = plt.figure()
    indices = np.random.choice(mispredict_indices, 4)
    for i in range(4):
        a = fig.add_subplot(2, 2, i+1)
        image = data[indices[i]]
        print('shape ',image.shape)
        image = DataProcessor.denormalize(image, 127.5, 255)
        image = image.squeeze()
        implot = plt.imshow(image, cmap = 'Greys_r')
        #print(predictions[indices[i]])
        predicted_output = np.array_str(predictions[indices[i]])
        label = "predicted:"+predicted_output+" actual:"+np.array_str(labels[indices[i]])
        print(label)
        label = predicted_output
        a.set_title(label)
    fig1 = plt.gcf()
    plt.show()
    if mispredict:
        fig1.savefig('mispredicts.png')
    else:
        fig1.savefig('correct_predicts.png')

def load_network(file_name):
    print("Parsing network architecture from {}",file_name)
    network = Network(file_name)
    network.set_input_size((image_width, image_height, 1))
    #num_labels = 10 for 0 to 9 plus one if the digit does not exist
    network.set_output_size(num_labels+1)
    cake = LayerCake(network)
    return cake, network

def test_network(cake, network, dataset):

    dataset.reformat(image_width, image_height, num_channels)
    gc.collect()
    print("Test set ", dataset.test_dataset.shape, dataset.test_labels.shape)

    predictions = cake.run_prediction(None, dataset.test_dataset)
    print("shape of predictions ", predictions.shape)
    return predictions

def accuracy(predictions, labels, plot, data, categorized=False):
    compare_outputs = (predictions == labels)
    individual_accuracy = [None]*6
    if not categorized:
        for i in range(6):
            individual_accuracy[i] = (100.0 * np.sum(compare_outputs[:,i].astype(int))/ predictions.shape[0])
    compare_outputs_sum = np.sum(compare_outputs, 1)
    compare_outputs_sum = compare_outputs_sum/6
    if plot:
        plot_mispredicts(data, predictions, compare_outputs_sum, labels) 
        plot_mispredicts(data, predictions, compare_outputs_sum, labels, 0) 
    final_accuracy = (100.0 * np.sum(compare_outputs_sum)
          / predictions.shape[0])
    return individual_accuracy, final_accuracy

def return_predictions(predictions, index, rand=False):
    preds = np.ndarray(shape=(4, 6), dtype=DataType.ndtype)
    max_i = 0
    max_prob = 0

    if rand:
        i = random.randint(0, 4)
        preds[0,0] = i+1
        for k in range(5):
            if k <= i:
                preds[0,k+1] = random.randint(0,9)
            else:
                preds[0,k+1] = -1
        return preds[0,]

    for i in range(4): 
        j = i+1
        preds[i,0] = j
        preds[i,5] = -1
        prob = predictions[0][index, j]
        for k in range(4):
            if k <= i:
                prob = prob*max(predictions[k+1][index,0:num_labels])
                preds[i,k+1] = np.argmax(predictions[k+1][index,0:num_labels])
            else:
                prob = prob*(predictions[k+1][index,num_labels])
                preds[i,k+1] = -1
        if prob > max_prob:
            max_prob = prob
            max_i = i
    return preds[max_i,]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--random', dest = 'random', action='store_true')
    parser.add_argument('-x', '--read_extra', dest = 'read_extra', action='store_true')
    parser.set_defaults(random=False)
    parser.set_defaults(read_extra=False)
    args = parser.parse_args()
    return args

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

    if args.read_extra:
        data_file = base_path+str(image_width)+'x'+str(image_height)+'/extra_svhn_data.pickle' 

    dataset = read_dataset(args.read_extra)

    #For number of digits and for each digit generate prediction from the learnt network
    dir_names = ['num_digits_networks', 'digit_1_networks', 'digit_2_networks', 'digit_3_networks', 'digit_4_networks'] 
    predictions = [None for i in range(5)]
    for digit, name in zip(range(5),dir_names):
        network_filename = name+"/svhn_network.old.json"
        num_labels = 10
        if digit == 0:
            num_labels = 4
        cake, network = load_network(network_filename)
        cake.valid_train_step = 1000
        label_index = digit
        predictions[digit] = test_network(cake, network, dataset)
    test_output = np.ndarray(shape=(dataset.test_labels.shape[0], 6), dtype=DataType.ndtype)
    
    #combine prediction according to their softmax probabilities
    for i in range(dataset.test_labels.shape[0]):
        test_output[i,] = return_predictions(predictions, i, rand=args.random)
    ind_acc, test_acc = accuracy(test_output, dataset.test_labels, 0, dataset.test_dataset)
    for i in range(6):
        print('accuracy for digit %d %f' %(i, ind_acc[i]))
    print("Test accuracy: %.1f%%" % test_acc)
