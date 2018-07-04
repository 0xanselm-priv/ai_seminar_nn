"""Data-Types: This file contains the code for the new data-types.
These will be used to create the neural network later on.

How to:
In the main function, first, the data from test_data.txt is retrived,
and the input as well as the output is put into np ndarray form
Afterwards you can construct a nn using the explanation provided.
with network_robert a constructor is called which builds the nn.
In the following for loop the input data is put in and that output is
compared to the target output. at the end the overall costs of this nn is shown.

to do:
apply sgd
apply backprob
celebrate

Please note, that in the console printed version the weight and bias
values are rounded by default. Internally they are stored as 16bit float.
we can certainly expand to 32 or 64 bit or even 128.
"""

__author__ = "Niels #1, Niels #2"
__copyright__ = "Copyright 2017/2018 – EPR-Goethe-Uni"
__credits__ = "If you would like to thank somebody \
              i.e. an other student for her code or leave it out"
__email__ = ""

import numpy as np
import network_class
import random
from tkinter import *
import matplotlib.pyplot as plt
import cv2
from tensorflow.examples.tutorials.mnist import input_data



class Main():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    training_images = mnist.train.images
    training_labels = mnist.train.labels

    mnist_net = network_class.Network([(784), (400), (10)],  weights=None, bias=None, activation_function="sigmoid", initilizer="xavier_sigmoid", dropout=0.0)

    # Take random and put back

    for ind in range(len(training_images)):
        print("------------------------------------------------------------",ind)
        x = np.matrix(training_images[ind]).transpose()
        y = np.matrix(training_labels[ind]).transpose()
        mnist_net.test_train_single(x, y)

    test_7 = np.array(mnist.test.images[0]).transpose()
    test_cl = np.array(mnist.test.images[1]).transpose()
    test_cl2 = np.array(mnist.test.images[2]).transpose()
    test_cl3 = np.array(mnist.test.images[3]).transpose()
    test_cl4 = np.array(mnist.test.images[4]).transpose()

    mnist_net.test_info(test_7, np.array(mnist.test.labels[0]).transpose())
    mnist_net.test_info(test_cl, np.array(mnist.test.labels[0]).transpose())
    mnist_net.test_info(test_cl2, np.array(mnist.test.labels[0]).transpose())
    mnist_net.test_info(test_cl3, np.array(mnist.test.labels[0]).transpose())
    mnist_net.test_info(test_cl4, np.array(mnist.test.labels[0]).transpose())




if __name__ == "__main__":
    Main()
