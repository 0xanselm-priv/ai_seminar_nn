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
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import skimage.measure
import time
import datetime
import threading


# ------------------------- Train a 196-100-10 Network with 10000 Images with Initialization random! --------------


class Main():
    def __init__(self):
        self.time_stamp = str(datetime.datetime.now()).replace(' ', '_').replace(':', '')[:-7]
        self.training_images = []
        self.training_labels = []
        self.test_images = []
        self.test_labels = []
        self.main()



    def main(self):
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# ----- MAX POOLING OF IMAGES (2,2) ------
        for img in mnist.train.images:
            self.training_images.append(np.matrix(skimage.measure.block_reduce(np.matrix(img).reshape((28,28)), (2,2), np.max)).flatten().transpose())

        for lab in mnist.train.labels:
            self.training_labels.append(np.matrix(lab).transpose())

        for img in mnist.test.images:
            self.test_images.append(np.matrix(skimage.measure.block_reduce(np.matrix(img).reshape((28,28)), (2,2), np.max)).flatten().transpose())

        for lab in mnist.test.labels:
            self.test_labels.append(np.matrix(lab).transpose())

# # ----- SHOW IMAGES OF NUMBERS ------
#
#         test1 = np.array(mnist.train.images[986]).reshape((28,28))
#         test1_down = self.training_images[986].reshape(14, 14)
#
#         print(self.training_labels[986])
#         img = plt.imshow(test1)
#         plt.show()
#         img = plt.imshow(test1_down)
#         plt.show()

        # self.params = self.read_params('start_weights_for_test.txt')


        percentage = 0

        start3 = datetime.datetime.now()
        start_time = datetime.datetime.now()

        mnist_net_single = network_class.Network([(196), (100), (10)],  weights=None, bias=None, activation_function="sigmoid", initilizer="random", dropout=0.0)

        acc1 = []
        acc2 = []

        for ind in range(10000):
            ind = random.randint(0, 55000)
            percentage = percentage + 1 / 10000
            if ind % 250 == 0:
                error1 = 0
                error2 = 0
                for i in range(10000):
                    x = mnist_net_single.test(self.test_images[i])
                    first = np.argmax(x)
                    seccond = np.argmax(np.delete(x, first))
                    if seccond >= first:
                        seccond += 1
                    y = np.argmax(self.test_labels[i])

                    if first != y:
                        error1 += 1
                        if seccond != y:
                            error2 += 1

                acc1.append(1 - error1 / 10000)
                acc2.append(1 - error2 / 10000)

            x = self.training_images[ind]
            y = self.training_labels[ind]
            mnist_net_single.test_train_single(x, y)

            print('\n' + str(percentage) + '%\n')

        mnist_net_single.save_params('weights_after_test_9')

        end3 = datetime.datetime.now()
        self.write_time(start3, end3, start_time)

        self.write_acc(acc1)
        self.write_acc(acc2)






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

    def write_time(self, start, end, start_time):
        title = 'times_for_mnist_net_9_' + self.time_stamp + '.txt'
        with open(title, 'a') as f:
            f.write('Time Differrence: ' + str(end - start) + '\n')
            f.write('TIME of Start' + str(start_time).replace(' ', '_').replace(':', '')[:-7] + '\n')

    def write_acc(self, list):
        title = 'accuracy_for_mnist_net_9_' + self.time_stamp + '.txt'
        with open(title, 'a') as f:
            f.write(str(list) + '\n')






if __name__ == "__main__":
    Main()
