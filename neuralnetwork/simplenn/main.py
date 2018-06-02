"""Data-Types: This file contains the code for the new data-types.

These will be used to create the neural network later on.
"""

import numpy as np
import math
import random
import sys
# Eclipse style imports. Inform me if this
# this bothers your IDE
import simplenn
from simplenn import network
from simplenn import node
from simplenn import network_robert

__author__ = "Niels #1, Niels #2"
__copyright__ = "Copyright 2017/2018 – EPR-Goethe-Uni"
__credits__ = "If you would like to thank somebody \
              i.e. an other student for her code or leave it out"
__email__ = "niels.heissel@stud.uni-frankfurt.de"


#
# a = np.mat('2; 3; 4; 5')
#
# print(a)


def main():
#     first_network = network.Network([[node.Node(1, 0), node.Node(2, 1)], [node.Node(0, 0)], [
#         node.Node(0, 0), node.Node(5, 1), node.Node(4, 2)]])
#     first_Network.create_network()
#     first_network.create_network_randomly()
    nn_conf = [(1,4),(2,3),(3,4),(4,5),(5,2)]
    a = network_robert.Network(nn_conf, "relu")

if __name__ == "__main__":
    main()
