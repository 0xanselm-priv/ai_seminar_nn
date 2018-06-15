"""Data-Types: This file contains the code for the new data-types.

These will be used to create the neural network later on.
"""

import numpy as np
import math
import random

__author__ = "6770541: Niels Heissel, 6785468: Robert am Wege"
__copyright__ = "Copyright 2017/2018 – EPR-Goethe-Uni"
__credits__ = "If you would like to thank somebody \
              i.e. an other student for her code or leave it out"
__email__ = "niels.heissel@stud.uni-frankfurt.de"



class Node(object):
    def __init__(self, bias, number):
        self.weight = []
        self.bias = bias
        self.name = number

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def activate(self):
        print("activated")

    def read_output(self, output):
        return output

    def add_weight(self, w):
        self.weight.append(w)



class Network(object):
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        # self.bias = bias

        self.in_layer = layers[0]

    def create_network(self):

        for i in range(len(self.layers)-1):
            weigth_string = ""
            for node in self.layers[i+1]:
                for pre_node in self.layers[i]:
                    w = input("Whats the nodes weight from " + str(pre_node.name) + " to " + str(node.name) + "?")
                    node.add_weight(w)
                    weigth_string += w + " "
                    print(w)
                weigth_string += "; "

            weigth_matrice = np.mat(weigth_string[:-2])
            self.weights.append(weigth_matrice)

    def create_randomw_network(self):
        for i in range(len(self.layers)-1):
            weigth_string = ""
            for node in self.layers[i+1]:
                for pre_node in self.layers[i]:
                    w = str(random.random())
                    node.add_weight(w)
                    weigth_string += w + " "
                    print(w)
                weigth_string += "; "

            weigth_matrice = np.mat(weigth_string[:-2])
            self.weights.append(weigth_matrice)

    def print_weights(self):
        print(self.weights)

    def run(self, x):
        in_put = np.mat(x)

        for i in range(len(self.layers)-1):
            try:
                in_put =  self.weights[i] * in_put
            except ValueError:
                print("Error")


        print(in_put)




First_Network = Network([[Node(1, 0), Node(2,1)], [Node(0, 0)], [Node(0,0), Node(5,1), Node(4,2)]])

First_Network.create_randomw_network()
First_Network.print_weights()
First_Network.run("1; 2")
