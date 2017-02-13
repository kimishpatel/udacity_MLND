from __future__ import print_function
import numpy as np
import tensorflow as tf
from sklearn import cross_validation as cv
from collections import deque
from deeplayer import *
from convlayer import *
import sys
sys.path.append('../capstone/')
from default_dtype import * 

def accuracy(predictions, labels, num_digits=6, num_labels=11):
    accuracy_array = np.ones((predictions[0].shape[0],1), dtype=bool)
    for i in range(num_digits):
        temp_arr = (np.argmax(predictions[i], 1) == np.argmax(labels[:,i*num_labels:(i+1)*num_labels], 1))
        temp_arr = temp_arr.reshape((predictions[0].shape[0],1))
        accuracy_array = np.logical_and(accuracy_array, temp_arr)
    return ((100.0 * np.sum(accuracy_array))
          / predictions[0].shape[0])

class LayerCake:
    def __init__(self, network):
        
        self.graph = tf.Graph()
        self.cv = 0
        self.session = None
        self.valid_train_step = 1000
        self.model_name = 'entire_number'
        with self.graph.as_default():
            #create placeholder for input data and output labels
            self.input_size = network.input_size
            self.output_size = network.output_size
            self.num_labels = self.output_size[1]
            if isinstance(self.input_size, tuple):
                self.input_data = tf.placeholder(DataType.dtype, shape=(None,) + self.input_size)
            else:
                self.input_data = tf.placeholder(DataType.dtype, shape=(None, self.input_size))
            self.input_labels = tf.placeholder(DataType.dtype, shape=(None, self.output_size[0]*self.output_size[1]))
            self.reg = network.reg
            self.dropout = network.dropout
            #store slices of the network in here
            self.slices = []

            self.max_steps_down = 10

            prev_layer = deque(maxlen=1)
            prev_layer.append(self.input_data)
            prev_layer_size = deque(maxlen=1)
            prev_layer_size.append(self.input_data.get_shape().as_list())
            for layer in network.layers:

                if layer['type'] == 'conv':
                    l = ConvLayer(self.graph, layer['layers'], layer['activations'], layer['pooling'], input_data = prev_layer[0])

                if layer['type'] == 'fc':
                    #output of a non fc layer: reshape it
                    prev_layer_input = prev_layer[0]
                    shape = prev_layer_size[0]
                    prev_size = shape
                    print(type(shape))
                    if isinstance(shape, list):
                        print(shape)
                        if isinstance(shape[0], int):
                            prev_layer_input = tf.reshape(prev_layer_input, [shape[0], shape[1]*shape[2]*shape[3]])
                        else:
                            prev_layer_input = tf.reshape(prev_layer_input, [-1, shape[1]*shape[2]*shape[3]])
                        prev_size = shape[1]*shape[2]*shape[3]
                        print('prev size', prev_size)
                    l = ConnectedLayer(self.graph, [prev_size] + layer['layers'], layer['activations'], input_data = prev_layer_input, dropout = self.dropout)
                prev_layer.append(l.output)
                prev_layer_size.append(l.output_size)
                self.slices.append(l)

            #create the final layer
            output_layer = network.output
            if output_layer['final'] == 'softmax' and output_layer['loss'] == 'cross_entropy':
                self.output_weights = [None]*self.output_size[0]
                self.output_biases = [None]*self.output_size[0]
                self.output = [None]*self.output_size[0]
                for i in range(self.output_size[0]):
                    self.output_weights[i] = tf.Variable(tf.truncated_normal([l.output_size, self.output_size[1]], stddev = 0.1))
                    self.output_biases[i] = tf.Variable(tf.zeros([self.output_size[1]]))
                    self.output[i] = (tf.nn.softmax_cross_entropy_with_logits((tf.matmul(prev_layer[0], self.output_weights[i]) + self.output_biases[i]), self.input_labels[:,i*self.num_labels:(i+1)*self.num_labels]))
                print("Creating softmax w cross entropy loss function")
                self.loss = tf.reduce_mean(tf.add_n(self.output))

                self.output = [None]*self.output_size[0]
                self.predict_tensors = [None]*self.output_size[0]
                self.train_predictions = [None]*self.output_size[0]
                for i in range(self.output_size[0]):
                    self.predict_tensors[i] = tf.nn.softmax((tf.matmul(prev_layer[0], self.output_weights[i]) + self.output_biases[i]))
                    self.train_predictions[i] = tf.nn.softmax((tf.matmul(prev_layer[0], self.output_weights[i]) + self.output_biases[i]))

            if self.loss != None and self.reg != None:
                for sl in self.slices:
                    self.loss += sl.regularize(self.reg)
                regs = tf.nn.l2_loss(self.output_weights)
                self.loss += self.reg * regs
            
    def set_optimizer(self, optimizer_type, learning_rate = 0.5, learning_rate_decay = None): 

        with self.graph.as_default():
            self.global_step = tf.Variable(0, trainable=False)
            if learning_rate_decay != None:
                self.learning_rate = tf.train.exponential_decay(learning_rate, self.global_step, 1000, learning_rate_decay, staircase=True) 
            else:
                self.learning_rate = learning_rate

            if optimizer_type == None or optimizer_type == 'SGD':
                self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss, global_step = self.global_step)
            elif optimizer_type == 'Momentum':
                self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.8).minimize(self.loss, global_step = self.global_step)
            elif optimizer_type == 'adam':
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step = self.global_step)
            else:
                print("Unrecognized optimizer")

    def run_prediction(self, sess, input_dataset, batch_size=None):
        if batch_size == None:
            feed_dict = {self.input_data : input_dataset}
        else:
            feed_dict = {self.input_data : input_dataset[0:batch_size]}
        if sess != None:
            predictions = [None]*self.output_size[0]
            for i in range(self.output_size[0]):
                predictions[i] = sess.run(self.predict_tensors[i], feed_dict = feed_dict)
            return predictions
        else:
            with tf.Session(graph=self.graph) as session:
                self.model_name = 'entire_number'
                saver = tf.train.Saver()
                ckpt = tf.train.get_checkpoint_state('./', self.model_name+'.ckpt') 
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(session, ckpt.model_checkpoint_path)
                predictions = self.calculate_predictions(session, input_dataset)
            return predictions

    def calculate_predictions(self, session, data):
        output_predictions = np.ndarray(shape=(data.shape[0], self.output_size[0]*self.output_size[1]))
        for i in range(0, data.shape[0], self.valid_train_step):
            if i+self.valid_train_step < data.shape[0]: 
                predictions = self.run_prediction(session, data[i:i+self.valid_train_step])
                for j in range(self.output_size[0]):
                    start_index = j*self.output_size[1]
                    end_index = start_index+self.output_size[1]
                    output_predictions[i:i+self.valid_train_step,start_index:end_index] = predictions[j][:,:]
            else:
                predictions = self.run_prediction(session, data[i:])
                for j in range(self.output_size[0]):
                    start_index = j*self.output_size[1]
                    end_index = start_index+self.output_size[1]
                    output_predictions[i:,start_index:end_index] = predictions[j][:,:]
        return output_predictions

    def calculate_accuracy(self, session, data, labels):
        num_batches = 0
        accu = 0.
        for i in range(0, data.shape[0], self.valid_train_step):
            if i+self.valid_train_step < data.shape[0]: 
               num_batches += 1
               predictions = self.run_prediction(session, data[i:i+self.valid_train_step])
               accu += accuracy(predictions, labels[i:i+self.valid_train_step])
            else:
               num_batches += 1
               predictions = self.run_prediction(session, data[i:])
               accu += accuracy(predictions, labels[i:])
        accu = (float(accu) / float(num_batches))
        return accu

    def run_training(self, num_steps, input_data, input_labels, valid_data, valid_labels, test_data, test_labels, batch_size=None, save_model = False):
        with tf.Session(graph=self.graph) as session:
            self.model_name = 'entire_number'
            if save_model:
                saver = tf.train.Saver()  
            self.session = session
            tf.initialize_all_variables().run()
            print("Initialized")
            best_val_accuracy = 0.
            prev_val_accuracy = 0.
            val_accuracy_down = 0
            for step in range(num_steps):
                # Pick an offset within the training data, which has been randomized.
                # Note: we could use better randomization across epochs.
                if batch_size == None:
                    batch_data = input_data
                    batch_labels = input_labels
                else:
                    if self.cv:
                        sss = cv.StratifiedShuffleSplit(np.argmax(input_labels, 1), train_size=batch_size, n_iter = 3)
                        for item in sss:
                            pass
                        batch_data = input_data[item[0][:batch_size], :]
                        batch_labels = input_labels[item[0][:batch_size], :]
                    else:
                        offset = (step * batch_size) % (input_labels.shape[0] - batch_size)
                        batch_data = input_data[offset:(offset + batch_size), :]
                        batch_labels = input_labels[offset:(offset + batch_size), :]
                feed_dict = {self.input_data : batch_data, self.input_labels : batch_labels}
                _, l= session.run(
                  [self.optimizer, self.loss], feed_dict=feed_dict)
                predictions = [None]*self.output_size[0]
                for i in range(self.output_size[0]):
                    predictions[i] = session.run(
                    self.train_predictions[i], feed_dict=feed_dict)
                if (step % 500 == 0):
                    print("Minibatch loss at step %d: %f" % (step, l))
                    print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                    validation_accuracy = self.calculate_accuracy(session, valid_data, valid_labels)
                    print("Validation accuracy: %.1f%%" % validation_accuracy)

                    if round(validation_accuracy, 2) > round(best_val_accuracy, 2):
                        best_val_accuracy = validation_accuracy
                        test_accuracy = self.calculate_accuracy(session, test_data, test_labels)
                        print("Test accuracy at step %d: %.1f%%" % (step, test_accuracy))
                        if save_model:
                            print("Saving session at step %d"% step)
                            saver.save(session, self.model_name, latest_filename=self.model_name+'.ckpt')

            test_accuracy = self.calculate_accuracy(session, test_data, test_labels)
            print("Test accuracy: %.1f%%" % test_accuracy)
            return test_accuracy
