import numpy as np
import random
import sys
import math


class Network():
    weights_matrices = []  # saves all the weight-matrices as list of matrices
    bias_matrices = []  # saves all bias vectors as list of vectors

    activation_vectors = []

    conf_list = 0  # what is this used for?
    layers_count = 0
    activation_funct = 0
    input_size = 0
    input_vector = 0
    output_vector = 0
    target_output = 0
    training_data_count = 0
    cost_function_sum = 0
    nn_cost_function = 0

    # number of layers should be cardinality of conf_list + 1
    # number of nodes per layer should be second value of said tuple

    def __init__(self, conf_list, activation_function, number_format=np.float16):
        if len(conf_list) < 2:
            print("Wrong or impossible Config. Check params")
            sys.exit()
        else:
            self.conf_list = conf_list
            self.activation_func = activation_function

            self.itterations = 500000

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
                self.bias_matrices.append(a)  # <----- import to use as real matrix (use np.asmatrix(matrix) !!!!

        # setting only possible input size dim
        self.input_size = self.weights_matrices[0].shape[1]
        self.nn_setup_rand()
        self.layers_count = len(self.weights_matrices) + 1
        self.nn_information()

        b = np.array(0, np.float16)
        b.resize((2, 1))
        x1 = 2
        x2 = 1
        b.itemset((0, 0), x1)
        b.itemset((1, 0), x2)

        self.apply_input(b)  # <---- applies input and propagates forwards
        self.target_output = np.array([[1],[0]])
        self.cost_function()

        #print("NOW WE UPDATE WEIGHTS AND BIASES!!!!!_________")

        for i in range(1, self.itterations):

            for layer_num in range(len(self.conf_list) - 1):
                print(layer_num + 2)
                self.update_w(layer_num + 2, 0.05)
                self.update_b(layer_num + 2, 0.05)

            self.apply_input(b)
            self.cost_function()

        self.save_outputs()

    def nn_setup(self):
        self.weights_matrices = [np.array([[0.4, 2],[-1.3, 0.1], [0.4, 0.5]]), np.array([[0.4, 0.3, 0.1], [0.4, 0.3, 0.1], [0.4, 0.3, 0.1], [0.4, 0.3, 0.1], [0.4, 0.3, 0.1]]), np.array([[0.6, 0.2, -0.9, 0.7, 1.4],[0.4, 1.1, 0.7, 1.3, 0.1]])]
        self.bias_matrices= [np.array([[0],[0.3],[2]]), np.array([[2], [0.3], [0.7], [-1.2], [3]]), np.array([[0.5],[0.6]])]

        # for matrix in self.weights_matrices:
        #     for m in range(matrix.shape[0]):
        #         for n in range(matrix.shape[1]):
        #             print(weight[m][n])
        #             matrix.itemset((m, n), weight[m][n])
        # for vector in self.bias_matrices:
        #     for m in range(vector.shape[0]):
        #         vector.itemset((m, 0), input("bias" + str(m)))

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
        print("Input Size:", self.input_size, "x 1")

    def get_activation(self, x):
        # obsolete idiot function. maybe dont delete yet
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
                a = (1 / (1 + np.exp(-a)))  # <------ could produce error
                matrix.itemset((m, 0), a)
        elif self.activation_func == "relu":
            for m in range(matrix.shape[0]):
                a = matrix.item((m, 0))
                a = np.maximum(m, 0)  # <------ produces error, dunno why tho
                matrix.itemset((m, 0), a)
        else:
            print("Error. Wrong Activation")
            sys.exit()
        return matrix

    def set_initial_randoms(self, a, b):
        # a and b for lower and upper bound of rand numbers
        for i in range(10):
            #             random.seed(4)
            real_nr = random.random() + 1
            int_nr = random.randint(a, b)
            real_nr = real_nr * int_nr
            return real_nr

    # def cost_function(self):
    #     self.training_data_count += 1
    #     # sub_to target and real outp
    #
    #     print(self.target_output)
    #     sub_to = np.subtract(self.target_output, self.output_vector)
    #     temp = np.linalg.norm(sub_to)
    #     temp = np.square(temp)
    #     temp = temp * (1 / 2)
    #     self.cost_function_sum = self.cost_function_sum + temp
    #
    #     print(self.cost_function_sum)

    def cost_function(self):
        print(self.target_output)
        cost = self.target_output - self.output_vector
        cost = np.linalg.norm(cost)
        cost = 1/2 * np.square(cost)
        print(cost)

        return cost

    def nn_cost(self):
        self.nn_cost_function = (
            1 / self.training_data_count) * self.cost_function_sum
        return (self.nn_cost_function)

    def propagate_forward(self):
        self.activation_vectors = []

        self.activation_vectors.append(self.input_vector)

        # x = 0
        # for i in self.weights_matrices:
        #     print(str(x), "\n")
        #     print(i)
        #     x += 1

        a = self.weights_matrices[0].dot(self.input_vector)  # is not a dot-product !!!! please use @-operator!!!! ?
        a = a + self.bias_matrices[0]
        a = self.vector_activator(a)  # <---- applies the activation function on vector

        self.activation_vectors.append(a)

        for i in range(1, self.layers_count - 1):
            #             print("Layer", i)
            #             print(a)

            a = self.weights_matrices[i].dot(a)  # is not a dot-product !!!! please use @-operator!!!! ?
            a = a + self.bias_matrices[i]
            a = self.vector_activator(a)

            self.activation_vectors.append(a)

#         print("Layer", self.layers_count - 1, "Output:\n", a)
        self.output_vector = a

        print("\n\n")
        print(self.output_vector)

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
#             print("Target Vector:\n", self.target_vector)

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
#             print("Input Vector\n", self.input_vector)
        # start propagating forward
        self.propagate_forward()

    def delta(self, layer):
        self.target_vector = np.array([[1],[0]])   # ---------> change this to be target output
        layer -= 1
        if layer + 1 == len(self.conf_list):
            d = self.activation_vectors[layer] * (1 - self.activation_vectors[layer]) * (self.activation_vectors[layer] - self.target_vector)  # <-- * = Hadamar Product

            return d
        else:
            return self.activation_vectors[layer] * (1 - self.activation_vectors[layer]) * (self.weights_matrices[layer].transpose() @ self.delta(layer + 2))

    def update_b(self, layer, learning_rate):
        for j in range(1, len(self.bias_matrices[layer - 2]) + 1):  # for every bias in that layer
                # calculate the corresponding bias-gradient and update vector
                self.bias_matrices[layer - 2][j-1] -= learning_rate * (self.delta(layer)[j-1])
        return self.delta(layer)

    def update_w(self, layer, learning_rate):

        for i in range(1, len(self.weights_matrices[layer - 2]) + 1):  # for every row in weight-matrix(layer)
            for j in range(1, len(self.weights_matrices[layer - 2][i - 1]) + 1):  # for every element in that row
                # calculate the corresponding weight-gradient and write as vector
                self.weights_matrices[layer - 2][i-1][j-1] -= learning_rate * (self.delta(layer)[i-1] * self.activation_vectors[layer - 2][j-1])

        # for matrix in self.weights_matrices:
        #     print("Matrix: ", matrix)

    def save_outputs(self):
        with open('ergebnisse.txt', 'a') as f:
            f.write(str(self.cost_function()) + "," + str(self.output_vector).replace('\n', ",") + "," + str(self.itterations) + "\n")

