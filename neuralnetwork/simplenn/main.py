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



class Main():

    def __init__(self):
        self.main()
        self.training_points = []
        self.dt = np.dtype("Float64")

    def main(self):
        dt = np.dtype("Float64")


# ----- Flower Example 2-1 Network
#
#          wei = [np.matrix([[0.9, 0.8]], dt)]
#          bia = [np.matrix([[1]], dt)]
#          a = network_class.Network([(2), (2), (1)], weights= wei, bias=bia, activation_function="sigmoid", initilizer="xavier_sigmoid")
#
#         data = [[3, 1.5, 1], [2, 1, 0], [4, 1.5, 1], [3, 1, 0], [3.5, .5, 1], [2, .5, 0], [5.5, 1, 1], [1, 1, 0]]
#         mysteriy = []
#
#         costs = []
#
#         for i in range(19999):
#             ind = np.random.randint(len(data))
#             point = data[ind]
#             a.test_train_single([[point[0]], [point[1]]], [point[2]])
#             costs.append(a.cost())
#
#         graph_x = []
#         graph_y = []
#
#         graph_x_r = []
#         graph_y_r = []
#
#         graph_x_b  = []
#         graph_y_b = []
#
#         for m in range(len(data)):
#             graph_x.append(data[m][0])
#             graph_y.append(data[m][1])
#
#         for y in range(len(graph_x)):
#             pred = a.test_info([[graph_x[y]],[graph_y[y]]], [data[y][2]])
#             if pred.item(0) > 0.5:
#                 graph_x_r.append(graph_x[y])
#                 graph_y_r.append(graph_y[y])
#             else:
#                 graph_x_b.append(graph_x[y])
#                 graph_y_b.append(graph_y[y])
#
#         plt.plot(graph_x_r, graph_y_r, 'ro')
#         plt.plot(graph_x_b, graph_y_b, 'bo')
#
#         plt.show()
#
#         plt.plot(costs)
#         plt.show()
#

# ----- Training a simple 1-1-1 Network
#         x = [[0.5]]
#         a.test_train_single(x, [[0]])
#         a.test_info(x, [[0]])

# ------ Training with 4 Inputs, 2 Hidden Layers, 1 Output
#         x = self.create_training_samples()
#
#         print("Hallo")
#         print(len(x))

        # for i in random.sample(x, 10000):
        #     a.test_train(i, self.create_target(i))
        #
        # a.print_nn_info()
        #
        # for i in range(20):
        #     y = self.create_batch(x)
        #     a.train_batch(y[0],y[1])
        #
        #
        # im = [[0.9], [0.2], [0.3], [0.0]]
        # aus = [[0.3], [0.5], [0.1], [0.5]]
        #
        # print("\n\n\n--------------")
        #
        # a.test_info(im, self.create_target(im))
        # a.test_info(aus, self.create_target(aus))

# #------ Binary 4-Digits --> ODD/EVEN
#
        b = network_class.Network([(4), (3), (1)], weights=None, bias=None, activation_function="sigmoid", initilizer="xavier_relu", dropout=0.0)

        data_b = []

        for first in range(2):
            for second in range(2):
                for third in range(2):
                    for fourth in range(2):
                        data_b.append([[first],[second],[third],[fourth]])

        print(len(data_b))

        x = random.sample(data_b, 14)  # <--- Take only a small amount of numbers train with these. 8 of 16 is very good!
        costs_b = []

        # # Training in Batch
        # for i in range(5000):
        #     tar = []
        #     y = random.sample(data_b, 4)
        #     for num in y:
        #         tar.append([eval("0b" + str(num[0][0]) + str(num[1][0]) + str(num[2][0]) + str(num[3][0])) % 2])
        #     b.train_batch(y, tar)
        #     costs_b.append(b.cost())

        # # TRAINING-SINGLE

        for i in range(10000):
            ind = np.random.randint(len(x))
            point = x[ind]
            b.test_train_single(point, [eval("0b" + str(point[0][0]) + str(point[1][0]) + str(point[2][0]) + str(point[3][0])) % 2])
            costs_b.append(b.cost())
        #
        values = []

        #params = self.read_params('test.txt')

        #test = network_class.Network(params[0], weights=params[1], bias=params[2], activation_function="sigmoid", initilizer="predefined")

        for data in data_b:
            values.append(b.test(data).item(0))

        plt.plot(costs_b)
        plt.show()

        plt.plot(values)
        plt.show()


# ------ Image creation
#         img_list = np.full((50, 20), '#ffffff')
#         img_list_2 = np.full((100, 100), '#ffffff')
#
#
#         img = cv2.imread('/Users/nielsheissel/Desktop.1.jpg', 1)
#         x = []
#         for m in range(50):
#             for y in range(20):
#                 x.append([[m/10],[y/10]])
#
#         for i in x:
#             output = a.test(i)
#             img_list[int(i[0][0]*10)][int(i[1][0]*10)] = output.item(0,0)
#             print(type(output.item(0,0)))
#
#         print(img)
#
#         plt.imshow(img)

# ---- Deutschland Karte Example
#
#         c = network_class.Network([(2), (4), (2)], weights=None, bias=None, activation_function="sigmoid", initilizer="xavier_relu", dropout=0.0)
#
#         image_set = self.data_fetch()
#
#         training_set = random.sample(image_set, 30000)
#
#         cost = []
#
#         for i in range(5000):
#             ind = np.random.randint(len(training_set))
#             point = training_set[ind]
#             c.test_train_single([[int(point[0]) / 50], [int(point[1]) / 50]], [[int(point[2])]])
#             cost.append(c.cost())
#
#         graph_x_r = []
#         graph_y_r = []
#
#         graph_x_b = []
#         graph_y_b = []
#
#         for i in range(5000):
#             point = random.sample(image_set, 1)[0]
#             value = c.test([[int(point[0]) / 50], [int(point[1]) / 50]])
#
#             if value > 0.5:
#                 graph_x_b.append(int(point[0]) / 50)
#                 graph_y_b.append(int(point[1]) / 50)
#             else:
#                 graph_x_r.append(int(point[0]) / 50)
#                 graph_y_r.append(int(point[1]) / 50)
#
#         plt.plot(graph_x_r, graph_y_r, 'ro')
#         plt.plot(graph_x_b, graph_y_b, 'bo')
#
#         plt.show()
#
#         plt.plot(cost)
#         plt.show()
#
#         graph_x_r = []
#         graph_y_r = []
#
#         graph_x_b = []
#         graph_y_b = []
#
#         for i in range(5000):
#             point = random.sample(image_set, 1)[0]
#
#             if point[2] == "1":
#                 graph_x_b.append(int(point[0]) / 50)
#                 graph_y_b.append(int(point[1]) / 50)
#             else:
#                 graph_x_r.append(int(point[0]) / 50)
#                 graph_y_r.append(int(point[1]) / 50)
#
#         plt.plot(graph_x_r, graph_y_r, 'ro')
#         plt.plot(graph_x_b, graph_y_b, 'bo')
#
#         plt.show()

    def create_batch(self, x):
        target = []
        batch = []
        for i in random.sample(x, 20):
            batch.append(i)
            target.append(self.create_target(i))
        return batch, target

    def create_training_samples(self):
        training_points = []
        for i in range(11):
            for j in range(11):
                for m in range(11):
                    for n in range(11):
                        training_points.append([[i / 10], [j / 10], [m / 10], [n / 10]])
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
        A = Tk()
        B = Canvas(A)
        B.place(x=0, y=0, height=50, width=20)
        for a in range(50):
            for b in range(20):
                B.create_line(a, b, a + 1, b + 1, fill=img_list[a][b])  # where pyList is a matrix of hexadecimal strings
        A.geometry("500x200")
        mainloop()

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
        return list_temp

    def read_params(self, file_name):
        with open(file_name, 'r') as params:
            rows = params.read().split('\n')
            layer_infos = eval(rows[0])
            wei = []
            bia = []
            for i in range(len(layer_infos) - 1):
                wei.append(np.matrix(eval(str(rows[i + 1]))))
            for i in range(len(layer_infos) - 1):
                bia.append(np.matrix(eval(str(rows[i + len(layer_infos)]))))

            return (layer_infos, wei, bia)



        
if __name__ == "__main__":
    Main()

