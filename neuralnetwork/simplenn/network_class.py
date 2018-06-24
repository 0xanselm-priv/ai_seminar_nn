import numpy as np
import math
import sys

# DEUTSCHLAND... DEUUUUUUUUTSCHLAND


class Network:

    def __init__(self, layer_infos, activation_function="sigmoid", weights=[], bias=[], initilizer="random"):
        self.activation_function = activation_function
        self.initializer = initilizer
        self.layer_infos = layer_infos
        self.layer_number = len(layer_infos)
        self.weights = []
        self.bias = []
        self.activation = []
        self.output = 0
        self.target = 0

        if self.initializer == "predefined":
            print("Setting up Network...")
            self.layer_init(layer_infos, weights, bias)
        else:
            self.initializers()

        self.print_nn_info()
    
    def print_initilizer(self):
        if self.initializer == "random":
            return("Random")
        elif self.initializer == "xavier_sigmoid":
            return("Xavier Sigmoid")
        elif self.initializer == "xavier_relu":
            return("Xavier Relu")
        elif self.initializer == "xavier_bengio":
            return("Xavier and Bengio")
        elif self.initializer == "xavier_tf":
            return("Xavier TensorFlow")
        elif self.initializer == "seed":
            return("Random with Seed")
        elif self.initializer == "predefined":
            return("Predefined Weights and Bias")
        else:
            print("Wrong Initializer")
            sys.exit()
    
    def initializers(self):
        i = 1
        if self.initializer == "random":            
            for layer_num in range(self.layer_number - 1):
                self.weights.append(np.matrix(np.random.randint(5, size=(self.layer_infos[layer_num + 1], self.layer_infos[layer_num]))))
                self.bias.append(np.matrix(np.random.randint(5, size=(self.layer_infos[layer_num + 1], 1))))
        elif self.initializer == "seed":            
            for layer_num in range(self.layer_number - 1):
                seed = 5 * i
                np.random.seed(seed)
                self.weights.append(np.random.randn(self.layer_infos[layer_num + 1], self.layer_infos[layer_num]) * np.sqrt(1 / self.layer_number))
                self.bias.append(np.matrix(np.random.randint(5, size=(self.layer_infos[layer_num + 1], 1))))
                i += 1
        elif self.initializer == "xavier_sigmoid":
            for layer_num in range(self.layer_number - 1):
                self.weights.append(np.random.randn(self.layer_infos[layer_num + 1], self.layer_infos[layer_num]) * np.sqrt(1 / self.layer_number))
                self.bias.append(np.matrix(np.random.randint(5, size=(self.layer_infos[layer_num + 1], 1))))
        elif self.initializer == "xavier_relu":
            for layer_num in range(self.layer_number - 1):
                self.weights.append(np.random.randn(self.layer_infos[layer_num + 1], self.layer_infos[layer_num]) * np.sqrt(2 / self.layer_number))
                self.bias.append(np.matrix(np.random.randint(5, size=(self.layer_infos[layer_num + 1], 1))))
        elif self.initializer == "xavier_bengio":
            for layer_num in range(self.layer_number - 1):
                self.weights.append(np.random.randn(self.layer_infos[layer_num + 1], self.layer_infos[layer_num]) * np.sqrt(2 / self.layer_number + self.layer_number + 1))
                self.bias.append(np.matrix(np.random.randint(5, size=(self.layer_infos[layer_num + 1], 1))))
        elif self.initializer == "xavier_tf":
            for layer_num in range(self.layer_number - 1):
                self.weights.append(np.random.randn(self.layer_infos[layer_num + 1], self.layer_infos[layer_num]) * np.sqrt(6 / self.layer_number + self.layer_number + 1))
                self.bias.append(np.matrix(np.random.randint(5, size=(self.layer_infos[layer_num + 1], 1))))
                
    def layer_init(self, layer_infos, weights, bias):
        print(weights)
        print(bias)
        if type(weights) == list and type(bias) == list:  # if weights are matrices
            for layer_num in range(self.layer_number - 1):
                # check whether weight-matrix is right shape
                if weights[layer_num].shape == (layer_infos[layer_num + 1], layer_infos[layer_num]):
                    self.weights.append(weights[layer_num])
                else:
                    print("Error in weight-matrix: Dimension ", weights[layer_num].shape)
            for layer_num in range(self.layer_number - 1):
                if bias[layer_num].shape == (layer_infos[layer_num + 1], 1):
                    self.bias.append(bias[layer_num])
                else:
                    print("Error in bias-matrix: Dimension ", bias[layer_num].shape)
        else:
            print("Wrong weight or bias format!")


    def print_nn_info(self):
        print("A CNN with " + str(self.layer_number) + " layers.")
        print("Activation Function used:", self.activation_function)
        print("Initializer used:", self.print_initilizer())
        for layer_num in range(self.layer_number):
            print("Layer", str(layer_num), "with", self.layer_infos[layer_num], "nodes")

        for i in range(len(self.bias)):
            layer_str = "Weight Matrix. Layer " + \
                str(i) + " -> " + str(i + 1) + " Matrix " + "\n"
            bias_str = "Layer " + str(i + 1) + " Bias Vector." + "\n"
            print(layer_str, self.weights[i])
            print(bias_str, self.bias[i])        
        print("Input Size:", self.layer_infos[0], "x 1")

    def activate(self, inp):
        if self.activation_function == "sigmoid":
            new = []
            for entry in inp:
                new.append([(1 / (1 + math.exp(-entry[0])))])
            return np.matrix(new)
        elif self.activate_function == "relu":
            # TO DO
            print("Not yet implemented")
            sys.exit()

    def propagate_forwards(self, inp):
        self.activation = [np.matrix(inp)]

        for layer in range(self.layer_number - 2):
            self.activation.append((self.activate(self.weights[layer] * self.activation[-1]) + self.bias[layer]))

        output = self.weights[-1] * self.activation[-1] + self.bias[-1]
        self.activation.append(output)
        return output

    def cost(self):
        cost = self.target - self.output
        cost = np.linalg.norm(cost)
        cost = 1 / 2 * np.square(cost)
        return cost

    def test(self, inp):
        print(self.propagate_forwards(inp))

    def test_info(self, inp, tar):
        self.target = np.matrix(tar)
        self.output = self.propagate_forwards(inp)
        cost = self.cost()
        print("\nInput: \n", np.matrix(inp), "\nTarget Output: \n", self.target, "\nOutput: \n", self.output, "\nCost: \n", cost)

    def delta(self, layer):
        if layer == self.layer_number - 1:
            print("DELTA FOR LAYER L ",np.multiply(np.multiply(self.activation[-1], (1 - self.activation[-1])), (self.output - self.target)))
            return np.multiply(np.multiply(self.activation[-1], (1 - self.activation[-1])), (self.output - self.target))  # self.output == self.activation[-1]
        else:
            return np.multiply(np.multiply(self.activation[layer], (1 - self.activation[layer])), np.transpose(self.weights[layer]) * self.delta(layer + 1))

    def update_b(self, eta):
        for layer in range(self.layer_number - 1):
            for entry in range(len(self.bias[layer])):
                self.bias[layer][entry] = self.bias[layer][entry] + eta * self.delta(layer)[entry]

    def update_w(self, eta):
        for layer in range(self.layer_number - 1):
            for entry in range(len(self.weights[layer])):
                for target in range(len(self.weights[layer][entry])):
                    self.weights[layer][entry][target] = self.weights[layer][entry][target] + eta * (self.delta(layer)[entry] * self.activation[layer][target])

    def backpropagation(self):
        eta = 0.05
        self.update_b(eta)
        self.update_w(eta)

    def test_train(self, inp, tar):
        print("TESTING BACKPROP!")
        self.test_info(inp, tar)
        self.backpropagation()
        self.test_info(inp, tar)
        self.print_nn_info()

