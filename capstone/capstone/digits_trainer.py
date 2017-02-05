# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
import sys
sys.path.append('../DNN/')
import matplotlib.pyplot as plt

from layercake import *
from network import *
import argparse
from data_processor import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_name', type=str, help='network json file', required=True)
    parser.add_argument('-s', '--skip_training', dest = 'training', action='store_false')
    parser.set_defaults(training=True)
    args = parser.parse_args()
    return args

pickle_file = 'digits_data.pickle'

image_size = 32
num_labels = 10
num_channels = 1 # grayscale

train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = DataProcessor.load_and_split_data('digits_data.pickle')

#Find edges. Apply edge filter
train_dataset = DataProcessor.apply_edge_filter(train_dataset)
valid_dataset = DataProcessor.apply_edge_filter(valid_dataset)
test_dataset = DataProcessor.apply_edge_filter(test_dataset)

#normalize data
train_dataset, valid_dataset, test_dataset = DataProcessor.normalize_data([train_dataset, valid_dataset, test_dataset], 127.5, 255)

def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels

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

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


args = parse_args()
print("Parsing network architecture from {}",args.file_name)
network = Network(args.file_name)
network.set_input_size((image_size, image_size, 1))
network.set_output_size(num_labels)
cake = LayerCake(network)
#batch_size = 16
batch_size = 32

#num_iterations = [20001, 30001, 50001]
num_iterations = [5001]
#batch_sizes = [16, 32, 64]
batch_sizes = [32]
plot_mispredicts = 0

for batch_size in batch_sizes:
    for num_iter in num_iterations:
        if args.training:
            cake.set_optimizer(network.optimizer, learning_rate = network.learning_rate, learning_rate_decay = 0.9)
            cake.max_steps_down = 30
            test_accuracy = cake.run_training(num_iter, train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels, batch_size)
        if plot_mispredicts:
            predictions = cake.run_prediction(None, test_dataset)
            #predictions = cake.run_prediction(cake.session, test_dataset)
            test_acc = accuracy(predictions, test_labels)
            print("Test accuracy: %.1f%%" % test_acc)
            plot_mispredicts(test_dataset, predictions, test_labels)

        with open('results.txt','a') as fd:
            fd.write("{}: {}\n".format(args.file_name, test_accuracy))
        fd.close()

