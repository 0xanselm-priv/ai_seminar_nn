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

class Main():
    input_vector_list = []
    target_vector_list = []
    data_list = []

    def __init__(self):
        self.main()

    def main(self):
        #     first_network = network.Network([[node.Node(1, 0), node.Node(2, 1)], [node.Node(0, 0)], [
        #         node.Node(0, 0), node.Node(5, 1), node.Node(4, 2)]])
        #     first_Network.create_network()
        #     first_network.create_network_randomly()
        #     nn_conf = [(1,4),(2,3),(3,4),(4,5),(5,2)] # Paper NN config
        #     control errything from here

        self.data_fetch()
        self.construct_input_vector()
        self.construct_target_vector_list()
        nn_conf_2x2 = [(1, 2), (2, 3), (3, 6), (4, 5), (5, 2)]
        a = network_robert.Network(nn_conf_2x2, "sigmoid")
        j = 0
        for i in self.input_vector_list:
            a.apply_input(i)            
            a.target_vector_constructor(self.target_vector_list[j])
            a.cost_function()
            j+= 1
        print(a.nn_cost())

    def data_fetch(self):
        fp_data = "test_data.txt"
        with open(fp_data, "r") as f:
            file_data = f.read()
        f.closed
        file_data = file_data.split(";")
        list_temp = []
        for i in range(len(file_data) - 1):
            separated = file_data[i].split("_")
            # m n category
            a = (separated[0], separated[1], separated[2])
            list_temp.append(a)
        self.data_list = list_temp

    def construct_input_vector(self):
        for i in self.data_list:
            a = np.array(0, np.float16)
            a.resize((2, 1))
            x1 = i[0]
            x2 = i[1]
            a.itemset((0, 0), x1)
            a.itemset((1, 0), x2)
            self.input_vector_list.append(a)

    def construct_target_vector_list(self):
        for i in self.data_list:
            a = np.array(0, np.float16)
            a.resize((2, 1))
            if i[2] == "0":

                a.itemset((0, 0), 1)
                a.itemset((1, 0), 0)
                self.target_vector_list.append(a)
            elif i[2] == "1":
                x1 = i[0]
                x2 = i[1]
                a.itemset((0, 0), 0)
                a.itemset((1, 0), 1)
                self.target_vector_list.append(a)


if __name__ == "__main__":
    Main()
