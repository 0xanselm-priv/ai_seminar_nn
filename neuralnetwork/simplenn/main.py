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

import numpy as np
import math
import random
import sys
import simplenn
from simplenn import network
from simplenn import node
from simplenn import network_robert

__author__ = "Niels #1, Niels #2"
__copyright__ = "Copyright 2017/2018 – EPR-Goethe-Uni"
__credits__ = "If you would like to thank somebody \
              i.e. an other student for her code or leave it out"
__email__ = "niels.heissel@stud.uni-frankfurt.de"


class Main():
    input_vector_list = []
    target_vector_list = []
    data_list = []

    def __init__(self):
        self.main()

    def main(self):
        #         first_network = network.Network([[node.Node(1, 0), node.Node(2, 1)], [node.Node(0, 0)], [
        #             node.Node(0, 0), node.Node(5, 1), node.Node(4, 2)]])
        #         first_Network.create_network()
        #         first_network.create_network_randomly()
        #         nn_conf = [(1,4),(2,3),(3,4),(4,5),(5,2)] # Paper NN config
        #         control errything from here

        self.data_fetch()
        self.construct_input_vector()
        self.construct_target_vector_list()
#         theese tuples are the structure of the network
#         each tuple represents a layer
#         the first entry is the layer number, starting with 1
#         because layer 0 is the input layer
#         and the second number describes the number of nodes/neurons
#         if you want to adjust the network just build a new tuple collection
        nn_conf_2x2 = [(1, 2), (2, 3), (3, 6), (4, 5), (5, 2)]
        # <----- could run with "relu". buggy rn
        a = network_robert.Network(nn_conf_2x2, "sigmoid")
        j = 0
#         for i in self.input_vector_list:
#             a.apply_input(i)
#             a.target_vector_constructor(self.target_vector_list[j])
#             a.cost_function()
#             j += 1
#         print(a.nn_cost())

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
