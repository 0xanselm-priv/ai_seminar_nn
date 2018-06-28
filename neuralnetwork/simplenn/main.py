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

class Main():

    def __init__(self):
        self.main()
        self.training_points = []
        self.dt = np.dtype("Float64")



    def main(self):
        dt = np.dtype("Float64")
        wei = [np.matrix([[3]], dt), np.matrix([[3]], dt)]
        #wei = [np.matrix([[0.5, .5], [0.5, 0.5], [0.5, 0.5]]), np.matrix([[0.3333, 0.3333, .3333], [0.3333, 0.33333, 0.33333]])]
        bia = [np.matrix([[1]], dt), np.matrix([[2]], dt)]
        #bia = [np.matrix([[0.33], [0.33], [0.33]]), np.matrix([[0.5], [0.5]])]
        a = network_class.Network([(1), (1), (1), (1)], weights= wei, bias=bia, activation_function="sigmoid", initilizer="xavier_sigmoid")

# ----- Training a simple 1-1-1 Network
        x = [[0.5]]
        a.test_train_single(x, [[0]])
        a.test_info(x, [[0]])

# ------ Training with 4 Inputs, 2 Hidden Layers, 1 Output
        # x = self.create_training_samples()
        #
        # print("Hallo")
        # print(len(x))
        #
        # for i in random.sample(x, 10000):
        #     a.test_train(i, self.create_target(i))
        #
        # a.print_nn_info()
        #
        #
        # im = [[0.9], [0.2], [0.3], [0.0]]
        # aus = [[0.3], [0.5], [0.1], [0.5]]
        #
        # print("\n\n\n--------------")
        #
        # a.test_info(im, self.create_target(im))
        # a.test_info(aus, self.create_target(aus))


# ------ Image creation
        # img_list = np.full((100, 100), '#ffffff')
        # img_list_2 = np.full((100, 100), '#ffffff')
        #
        # for i in x:
        #     output = a.test(i)
        #     img_list[i[0][0]][i[1][0]] = self.color_set(output.item(0))
        #     img_list_2[i[0][0]][i[1][0]] = self.color_set(output.item(1))
        #
        # self.draw_image(img_list)
        # self.draw_image(img_list_2)

    def create_training_samples(self):
        training_points = []
        for i in range(11):
            for j in range(11):
                for m in range(11):
                    for n in range(11):
                        training_points.append([[i/10], [j/10], [m/10], [n/10]])
        return training_points

    def create_target(self, inp):
        if inp[3][0] == 0.0:
            return [[1]]
        else:
            return [[0]]

    def greyscale_in_rgb(self, greyscale_black):
        return (255 - 255 * greyscale_black, 255 - 255 * greyscale_black, 255 - 255 * greyscale_black)

    def rgb_in_hex(self_, rgb):
         return "#%02x%02x%02x" % (int(rgb[0]), int(rgb[1]), int(rgb[2]))

    def color_set(self, greyscale):
        return self.rgb_in_hex(self.greyscale_in_rgb(greyscale))

    def draw_image(self, img_list):
        A=Tk()
        B=Canvas(A)
        B.place(x=0,y=0,height=100,width=100)
        for a in range(100):
            for b in range(100):
                B.create_line(a,b,a+1,b+1,fill=img_list[a][b])  # where pyList is a matrix of hexadecimal strings
        A.geometry("171x207")
        mainloop()




        
if __name__ == "__main__":
    Main()

