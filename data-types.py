"""Data-Types: This file contains the code for the new data-types.

These will be used to create the neural network later on.
"""

import numpy as np
import math

__author__ = "Niels #1, Niels #2"
__copyright__ = "Copyright 2017/2018 – EPR-Goethe-Uni"
__credits__ = "If you would like to thank somebody \
              i.e. an other student for her code or leave it out"
__email__ = "niels.heissel@stud.uni-frankfurt.de"


a = np.mat('2; 3; 4; 5')

print(a)


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
        # self.weights = weights
        # self.bias = bias

        self.in_layer = layers[0]

    def create_network(self):
        for i in range(len(self.layers)-1):
            #weight_matrix = np.mat(1 0 0; )
            weigth_string = ""
            for node in self.layers[i+1]:
                for pre_node in self.layers[i]:
                    w = input("Whats the nodes weight from " + str(pre_node.name) + " to " + str(node.name) + "?")
                    node.add_weight(w)
                    weigth_string += w + "; "
                    print(w)

            print(weigth_string[:-2])
            weigth_matrice = np.mat(weigth_string[:-2])
            print(weigth_matrice)






first_Network = Network([[Node(1, 0), Node(2,1)], [Node(0, 0)], [Node(0,0), Node(5,1), Node(4,2)]])

first_Network.create_network()
