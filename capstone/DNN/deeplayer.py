from __future__ import print_function
import numpy as np
import tensorflow as tf
from sklearn import cross_validation as cv
from collections import deque


cv = 0
def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

class ConnectedLayer:
    def __init__(self, graph, layers, activation_functions, dropout = None, input_data = None):
        self.layer_type = 'fc'
        self.graph = graph
        self.weights = [None] * (len(layers)-1)
        self.biases = [None]*(len(layers)-1)
        self.logits = [None]*(len(layers)-1)
        self.predict_tensor = None
        self.input_data = input_data
        with self.graph.as_default():
            
            prev_layer = deque(maxlen=1)
            prev_layer.append(self.input_data)
            for i in range(len(layers)-1):
                #weights and biases for layer i
                print("layer {} {}x{}".format(i, layers[i], layers[i+1]))
                self.weights[i] = tf.Variable(tf.truncated_normal([layers[i], layers[i+1]], stddev = 0.1))
                self.biases[i] = tf.Variable(tf.zeros([layers[i+1]]))
                
                #Now create operations for each layer using appropriate activation functions
                if i < len(activation_functions) and activation_functions[i] == 'relu':
                    self.logits[i] = tf.nn.relu((tf.matmul(prev_layer[0], self.weights[i]) + self.biases[i]))
                elif i < len(activation_functions) and activation_functions[i] == 'relu6':
                    self.logits[i] = tf.nn.relu6((tf.matmul(prev_layer[0], self.weights[i]) + self.biases[i]))
                elif i < len(activation_functions) and activation_functions[i] == 'softmax':
                    self.logits[i] = tf.nn.softmax((tf.matmul(prev_layer[0], self.weights[i]) + self.biases[i]))
                elif  i < len(activation_functions) and activation_functions[i] == 'linear':
                    self.logits[i] = tf.matmul(prev_layer[0], self.weights[i] + self.biases[i])
                else: #Default is tanh
                    self.logits[i] = tf.tanh(tf.matmul(prev_layer[0], self.weights[i] + self.biases[i]))

                if dropout != None:
                    self.logits[i] = tf.nn.dropout(self.logits[i], dropout)
                
                #push this layer into deque to reference to in the next one
                prev_layer.append(self.logits[i])

            self.output = self.logits[-1]
            self.output_size = layers[-1]

    def regularize(self, regularization):
        regs = tf.nn.l2_loss(self.weights[0])
        for weight in self.weights[1:]:
            regs += tf.nn.l2_loss(weight)
        regs = regularization * regs
        return regs
                                                                             
