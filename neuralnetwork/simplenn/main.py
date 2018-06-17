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
#import simplenn
#from simplenn import network
#from simplenn import node
import network_robert
from tkinter import *

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

        b = np.array(0, np.float16)
        b.resize((2, 1))
        x1 = 1
        x2 = 0
        b.itemset((0, 0), x1)
        b.itemset((1, 0), x2)

        nn_conf_2x2 = [(1, 2), (2, 3), (3, 4), (4, 2)]
        # <----- could run with "relu". buggy rn
        a = network_robert.Network(nn_conf_2x2, "sigmoid")

        a.train_nn(b, np.array([[1],[0]]))


        for i in random.sample(range(len(self.input_vector_list)), 200):
            print(self.target_vector_list[i])
            print(self.input_vector_list[i])
            a.train_nn(self.input_vector_list[i], self.target_vector_list[i])



        a.test(self.input_vector_list[17])
        a.test(self.input_vector_list[1002])
        a.test(self.input_vector_list[102])
        a.test(self.input_vector_list[20000])
        a.test(self.input_vector_list[999])

        a.nn_information()

        # img_list = np.full((171, 207), '#ffffff')
        #
        #
        #
        # counter = 0
        # print(a.test(self.input_vector_list[2]))
        # for i in range(171):
        #     for j in range(207):
        #         print(self.input_vector_list[counter])
        #         img_list[i][j] = self.color_set(a.test(self.input_vector_list[counter])[1])
        #         counter += 1
        #self.draw_image(img_list)

        # j = 0
        # for i in self.input_vector_list:
        #     a.apply_input(i)
        #     a.target_vector_constructor(self.target_vector_list[j])
        #     a.cost_function()
        #     j += 1
        # print(a.nn_cost())

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

    def greyscale_in_rgb(self, greyscale_black):
        return (255 - 255 * greyscale_black, 255 - 255 * greyscale_black, 255 - 255 * greyscale_black)

    def rgb_in_hex(self_, rgb):
         return "#%02x%02x%02x" % (int(rgb[0]), int(rgb[1]), int(rgb[2]))

    def color_set(self, greyscale):
        return self.rgb_in_hex(self.greyscale_in_rgb(greyscale))

    def draw_image(self, img_list):
        A=Tk()
        B=Canvas(A)
        B.place(x=0,y=0,height=171,width=207)
        for a in range(171):
            for b in range(207):
                B.create_line(a,b,a+1,b+1,fill=img_list[a][b])#where pyList is a matrix of hexadecimal strings
        A.geometry("171x207")
        mainloop()


if __name__ == "__main__":
    Main()



