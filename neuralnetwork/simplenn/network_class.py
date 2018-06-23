import numpy as np
import random
import sys
import math


class Network:
    def __init__(self, layer_infos, activation_function, weights = [], bias = []):
        self.activation_function = activation_function
        self.layer_infos = layer_infos
        self.layer_number = len(layer_infos)
        self.weights = []
        self.bias = []
        self.activation = []

        self.layer_init(layer_infos, weights, bias)
        self.print_nn_info()





    def layer_init(self, layer_infos, weights, bias):
        if type(weights) == np.matrixlib.defmatrix.matrix and type(bias) == np.matrixlib.defmatrix.matrix:  # if weights are matrices
            for layer_num in range(self.layer_number):
                # check wether weight-matrix is right shape
                if weights[layer_num].shape == (layer_infos[layer_num], layer_infos[layer_num + 1]):
                    self.weights.append(weights[layer_num])
                else:
                    print("Sorry wrong dimension for your weight-matrix: Dimension ", weights[layer_num].shape)
                if bias[layer_num].shape == (layer_infos[layer_num], 1):
                    self.bias.append(bias[layer_num])
                else:
                    print("Sorry wrong dimension for your bias-matrix: Dimension ", bias[layer_num].shape)

        else:
            print("You choose to randomly assign weights and bias.")
            for layer_num in range(self.layer_number - 1):
                self.weights.append(np.matrix(np.random.randint(5, size=(layer_infos[layer_num + 1], layer_infos[layer_num]))))

            for neuron_num in layer_infos:
                self.bias.append(np.matrix(np.random.randint(5, size=(neuron_num, 1))))


    def print_nn_info(self):
        print("A CNN with " + str(self.layer_number) + " layers.")
        for layer_num in range(self.layer_number):
            print("Layer", str(layer_num), "with", self.layer_infos[layer_num], "nodes")
        for i in range(len(self.bias) - 1):
            layer_str = "Weight Matrix. Layer " + \
                str(i) + " -> " + str(i + 1) + " Matrix " + "\n"
            bias_str = "Layer " + str(i) + " Bias Vector." + "\n"
            print(layer_str, self.weights[i])
            print(bias_str, self.bias[i])
        print("Activation Function used:", self.activation_function)
        print("Input Size:", self.layer_infos[0], "x 1")

    def activate(self, inp):
        if self.activation_function == "sigmoid":
            new = []
            for entry in inp:
                new.append([(1 / (1 + math.exp(-entry[0])))])
            return np.matrix(new)


    def propagate_forwards(self, inp):
        self.activation = [[np.matrix(inp)]]
        for layer in range(self.layer_number - 1):
            self.activation.append(self.weights[layer] * self.activation[-1] + self.bias[layer])





A = Network([(3),(2),(2)], "sigmoid")


