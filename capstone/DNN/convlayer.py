from __future__ import print_function
import numpy as np
import tensorflow as tf
from sklearn import cross_validation as cv
from collections import deque
from math import ceil


cv = 0
def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

class ConvLayer:
    def __init__(self, graph, layers, activation_functions, pooling, dropout = None, input_data = None):
        self.layer_type = 'conv'
        self.graph = graph
        self.weights = [None] * (len(layers))
        self.biases = [None]*(len(layers))
        self.layers = [None]*(len(layers))
        self.predict_tensor = None
        self.input_data = input_data

        #These shall be input as well
        stride = [1, 1, 1, 1]
        max_pool_ksize = [1, 2, 2, 1]
        max_pool_stride = [1, 2, 2, 1]
        #Assume a 4D input tensor
        if len(input_data.get_shape().as_list()) != 4:
            print("Exactly 4D inputs and tensors are supported by this class")
            exit(-1)
        input_x = input_data.get_shape().as_list()[1]
        input_y = input_data.get_shape().as_list()[2]
        padding = 'SAME'
        with self.graph.as_default():
            
            prev_layer = deque(maxlen=1)
            prev_layer.append(self.input_data)
            for i in range(len(layers)):
                #weights and biases for layer i
                #each layer should be these following 4 attributes
                #patch_h patch_w input channels num filters
                print("layer {} {}".format(i, layers[i]))
                self.weights[i] = tf.Variable(tf.truncated_normal(layers[i], stddev = 0.1))
                self.biases[i] = tf.Variable(tf.zeros([layers[i][-1]]))
                
                #Now create filter
                conv = tf.nn.conv2d(prev_layer[0], self.weights[i], stride, padding = padding) 
                output_x = input_x
                output_y = input_y
                if padding == 'SAME':
                    output_x = (input_x/stride[1])
                    output_y = (input_y/stride[2])
                if pooling[i] == 'max':
                    conv_pool = tf.nn.max_pool(conv, max_pool_ksize, max_pool_stride, padding=padding)
                elif pooling[i] == 'avg':
                    conv_pool = tf.nn.avg_pool(conv, max_pool_ksize, max_pool_stride, padding=padding)
                else:
                    conv_pool = conv
                if padding == 'SAME' and not pooling[i] == 'none':
                    output_x = int(ceil(float(output_x)/float(max_pool_stride[1])))
                    output_y = int(ceil(float(output_y)/float(max_pool_stride[2])))
                if activation_functions[i] == 'relu':
                    self.layers[i] = tf.nn.relu(conv_pool + self.biases[i])
                elif activation_functions[i] == 'relu6':
                    self.layers[i] = tf.nn.relu6(conv_pool + self.biases[i])
                else: #default tanh
                    self.layers[i] = tf.tanh(conv_pool + self.biases[i])

                #sizes for the next layer
                input_x = output_x
                input_y = output_y
                
                if dropout != None:
                    self.logits[i] = tf.nn.dropout(self.logits[i], dropout)
                
                #push this layer into deque to reference to in the next one
                prev_layer.append(self.layers[i])

            self.output = self.layers[-1]
            print(self.output.get_shape().as_list())
            print('=>', output_y, '=>', output_x)
            if self.output.get_shape().as_list()[0] == None:
                first_dim = None
            else:
                first_dim = self.output.get_shape().as_list()[0]
            self.output_size = [first_dim, output_x, output_y, self.output.get_shape().as_list()[3]]

    def regularize(self, regularization):
        regs = tf.nn.l2_loss(self.weights[0])
        for weight in self.weights[1:]:
            regs += tf.nn.l2_loss(weight)
        regs = regularization * regs
        return regs
