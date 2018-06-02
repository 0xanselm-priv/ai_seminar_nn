import numpy as np
import random
import sys
import math


class Network():
    weights_matrices = []
    bias_matrices = []
    conf_list = 0
    activation_funct = 0
    input_size = 0
    input_vector = 0
    # number of layers should be cardinality of conf_list
    # number of nodes per layer should be second value of said tuple

    def __init__(self, conf_list, activation_function):
        if len(conf_list) < 2:
            print("Wrong Config. Check params")
            sys.exit()
        else:
            self.conf_list = conf_list
            self.activation_func = activation_function
            for i in range(len(conf_list) - 1):
                m = conf_list[i + 1][1]
                n = conf_list[i][1]
                a = np.array(0, np.float16)
                a.resize((m, n))
                self.weights_matrices.append(a)
            for i in range(1, len(conf_list)):
                n = conf_list[i][1]
                a = np.array(0, np.float16)
                a.resize((n, 1))
                self.bias_matrices.append(a)
        # self.bias_matrices[1].item(1, 0))
        # setting only possible input size dim
        self.input_size = self.weights_matrices[0].shape[1]
        self.nn_setup_rand()
        self.nn_information()

    def nn_setup_rand(self):
        # setting up the network with random real numbers
        # shape returns the m x n dim of a matrix
        for matrix in self.weights_matrices:
            for m in range(matrix.shape[0]):
                for n in range(matrix.shape[1]):
                    matrix.itemset((m, n), self.set_initial_randoms(-5, 5))
        for vector in self.bias_matrices:
            for m in range(vector.shape[0]):
                vector.itemset((m, 0), self.set_initial_randoms(-5, 5))

    def nn_information(self):
        count_layers = len(self.conf_list)
        nn_str = "A CNN with " + str(count_layers) + " layers."
        print(nn_str)
        for element in self.conf_list:
            print("Layer", element[0], "with", element[1], "nodes")
        for i in range(len(self.conf_list) - 1):
            layer_str = "Weight Matrix. Layer " + \
                str(i + 1) + " -> " + str(i + 2) + " Matrix " + "\n"
            bias_str = "Layer " + str(i + 1) + " Bias Vector." + "\n"
            print(layer_str, self.weights_matrices[i])
            print(bias_str, self.bias_matrices[i])
        print("Activation Function used:", self.activation_func)
        print("Input Size:", self.input_size)
        print("Input Vector:\n", self.input_vector)

    def get_activation(self, x):
        if self.activation_func == "sigmoid":
            return (1 / (1 + math.exp(-x)))
        elif self.activation_func == "relu":
            return(np.maximum(x, 0))
        else:
            print("Error.Wrong Activation")
            sys.exit()

    def set_initial_randoms(self, a, b):
        # a and b for lower and upper bound of rand numbers
        for i in range(10):
            # random.seed(4)
            real_nr = random.random() + 1
            int_nr = random.randint(a, b)
            real_nr = real_nr * int_nr
            real_nr = round(real_nr, 3)
            return real_nr

    def apply_input(self, inp):
        # interpreting the n dim input tuple
        # and integrate it into a np data structure
        m = len(inp)
        if (m != self.input_size):
            print("Input vector size wrong. Error!")
            sys.exit()
        else:
            self.input_vector = np.array(0, np.float16)
            self.input_vector.resize((m, 1))
            for i in range(len(inp)):
                self.input_vector.itemset(i, inp[i])
            print("Input Vector\n", self.input_vector)
