import numpy as np
import random
import sys
import math


class Network():
    weights_matrices = []
    bias_matrices = []
    conf_list = 0
    layers_count = 0
    activation_funct = 0
    input_size = 0
    input_vector = 0
    output_vector = 0
    target_output = 0
    training_data_count = 0
    cost_function_sum = 0
    nn_cost_function = 0

    # number of layers should be cardinality of conf_list
    # number of nodes per layer should be second value of said tuple

    def __init__(self, conf_list, activation_function, number_format=np.float16):
        if len(conf_list) < 2:
            print("Wrong Config. Check params")
            sys.exit()
        else:
            self.conf_list = conf_list
            self.activation_func = activation_function
            for i in range(len(conf_list) - 1):
                m = conf_list[i + 1][1]
                n = conf_list[i][1]
                a = np.array(0, number_format)
                a.resize((m, n))
                self.weights_matrices.append(a)
            for i in range(1, len(conf_list)):
                n = conf_list[i][1]
                a = np.array(0, number_format)
                a.resize((n, 1))
                self.bias_matrices.append(a)
        # self.bias_matrices[1].item(1, 0))
        # setting only possible input size dim
        self.input_size = self.weights_matrices[0].shape[1]
        self.nn_setup_rand()
        self.layers_count = len(self.weights_matrices) + 1
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
        # Printing useful network information
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

    def get_activation(self, x):
        if self.activation_func == "sigmoid":
            return (1 / (1 + math.exp(-x)))
        elif self.activation_func == "relu":
            return(np.maximum(x, 0))
        else:
            print("Error.Wrong Activation")
            sys.exit()

    def vector_activator(self, matrix):
        if self.activation_func == "sigmoid":
            for m in range(matrix.shape[0]):
                a = matrix.item((m, 0))
                a = (1 / (1 + math.exp(-a)))
                matrix.itemset((m, 0), a)
        elif self.activation_func == "relu":
            for m in range(matrix.shape[0]):
                a = matrix.item((m, 0))
                a = np.maximum(m, 0)
                matrix.itemset((m, 0), a)
        else:
            print("Error.Wrong Activation")
            sys.exit()
        return matrix

    def set_initial_randoms(self, a, b):
        # a and b for lower and upper bound of rand numbers
        for i in range(10):
            # random.seed(4)
            real_nr = random.random() + 1
            int_nr = random.randint(a, b)
            real_nr = real_nr * int_nr
            return real_nr

    def cost_function(self):
        self.training_data_count += 1
        # sub_to target and real outp
        sub_to = np.subtract(self.target_output, self.output_vector)
        temp = np.linalg.norm(sub_to)
        temp = np.square(temp)
        temp = temp * (1 / 2)
        self.cost_function_sum = self.cost_function_sum + temp

    def nn_cost(self):
        self.nn_cost_function = (
            1 / self.training_data_count) * self.cost_function_sum
        return (self.nn_cost_function)

    def propagate_forward(self):
        a = self.weights_matrices[0].dot(self.input_vector)
        a = a + self.bias_matrices[0]
        a = self.vector_activator(a)
        for i in range(1, self.layers_count - 1):
            print("Layer", i)
            print(a)
            a = self.weights_matrices[i].dot(a)
            a = a + self.bias_matrices[i]
            a = self.vector_activator(a)
        print("Layer", self.layers_count - 1, "Output:\n", a)
        self.output_vector = a

    def target_vector_constructor(self, target_output):
        m = len(target_output)
        last_layer_mat_index = len(self.weights_matrices) - 1
        output_dim = self.weights_matrices[last_layer_mat_index].shape[0]
        if (m != output_dim):
            print("Target vector dimension wrong. Error!")
            sys.exit()
        else:
            self.target_vector = np.array(0, np.float16)
            self.target_vector.resize((m, 1))
            for i in range(len(target_output)):
                self.target_vector.itemset(i, target_output[i])
            print("Target Vector:\n", self.target_vector)

    def apply_input(self, inp):
        # interpreting the n dim input tuple
        # and integrate it into a np data structure
        m = len(inp)
        if (m != self.input_size):
            print("Input vector dimension wrong. Error!")
            sys.exit()
        else:
            self.input_vector = np.array(0, np.float16)
            self.input_vector.resize((m, 1))
            for i in range(len(inp)):
                self.input_vector.itemset(i, inp[i])
            print("Input Vector\n", self.input_vector)
        # start propagating forward
        self.propagate_forward()
