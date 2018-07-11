import numpy as np
import math
import sys
import matplotlib.pyplot as plt
import datetime


class Network:
    """Network:

    This class is a framework for all who want to use a neural-network for thejr problems.
    It allows for many individual tweeks and options for ones own network.

    For further help read the documentation.
    """

    def __init__(self, layer_infos, activation_function="sigmoid", weights=[], bias=[], initilizer="random", activate_all_layers=True, dropout=0.0):
        self.activate_all_layers = activate_all_layers
        self.activation_function = activation_function
        self.initializer = initilizer
        self.layer_infos = layer_infos
        self.layer_number = len(layer_infos)
        self.weights = []
        self.bias = []
        self.activation = []
        self.output = 0
        self.target = 0
        self.gradient_v = 0
        self.iteration = 1
        self.dt = np.dtype("Float64")
        self.w1 = []
        self.functions_calls = 0
        self.calculations = 0
        self.operations = 0
        self.dropout = dropout
        self.weights_old = self.weights
        self.dropout_matrices = []

        self.initializers(weights, bias)

        self.print_nn_info()

        # self.visualizer = data_visualization.DataVis(self.layer_infos) --- not needed at the moment

    def print_initilizer(self):
        """Prints the chosen initialization method."""
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
    
    def initializers(self, weights, bias):
        """Initializer:

        This method sets the weights for every layer, depending on the desired initialization method.
        You can choose the following methods: (random, seed, xavier_sigmoid, xavier_relu, xavier_bengio,
        xavier_tf, predefined). Please look at the Documentation of TF for overview on all methods.

        :parameter
            weights:: if predefined is the chosen method, this parameter holds the predefined weights
            bias:: if predefined is the chosen method, this parameter holds the predefined bias
        """
        i = 1
        if self.initializer == "random":            
            for layer_num in range(self.layer_number - 1):
                self.weights.append(np.matrix(np.random.randint(5, size=(self.layer_infos[layer_num + 1], self.layer_infos[layer_num]))))
                self.bias.append(np.matrix(np.random.randint(5, size=(self.layer_infos[layer_num + 1], 1))))
        elif self.initializer == "seed":            
            for layer_num in range(self.layer_number - 1):
                seed = 5 * i
                np.random.seed(seed)
                self.weights.append(np.random.randn(self.layer_infos[layer_num + 1], self.layer_infos[layer_num]) * np.sqrt(1 / self.layer_infos[layer_num]))
                self.bias.append(np.matrix(np.random.randint(5, size=(self.layer_infos[layer_num + 1], 1))))
                i += 1
        elif self.initializer == "xavier_sigmoid":
            for layer_num in range(self.layer_number - 1):
                self.weights.append(np.random.randn(self.layer_infos[layer_num + 1], self.layer_infos[layer_num]) * np.sqrt(1 / self.layer_infos[layer_num]))
                self.bias.append(np.matrix(np.random.randint(5, size=(self.layer_infos[layer_num + 1], 1)), self.dt))
        elif self.initializer == "xavier_relu":
            for layer_num in range(self.layer_number - 1):
                self.weights.append(np.random.randn(self.layer_infos[layer_num + 1], self.layer_infos[layer_num]) * np.sqrt(2 / self.layer_infos[layer_num]))
                self.bias.append(np.matrix(np.random.randint(5, size=(self.layer_infos[layer_num + 1], 1))))
        elif self.initializer == "xavier_bengio":
            for layer_num in range(self.layer_number - 1):
                self.weights.append(np.random.randn(self.layer_infos[layer_num + 1], self.layer_infos[layer_num]) * np.sqrt(2 / self.layer_infos[layer_num] + self.layer_infos[layer_num + 1]))
                self.bias.append(np.matrix(np.random.randint(5, size=(self.layer_infos[layer_num + 1], 1))))
        elif self.initializer == "xavier_tf":
            for layer_num in range(self.layer_number - 1):
                self.weights.append(np.random.randn(self.layer_infos[layer_num + 1], self.layer_infos[layer_num]) * np.sqrt(6 / self.layer_infos[layer_num] + self.layer_infos[layer_num + 1]))
                self.bias.append(np.matrix(np.random.randint(5, size=(self.layer_infos[layer_num + 1], 1))))
        elif self.initializer == "predefined":
            if type(weights) == list and type(bias) == list and len(weights) == self.layer_number - 1:  # if weights are matrices
                for layer_num in range(self.layer_number - 1):
                    # check whether weight-matrix is right shape
                    if weights[layer_num].shape == (self.layer_infos[layer_num + 1], self.layer_infos[layer_num]):
                        self.weights.append(weights[layer_num])
                    else:
                        print("Error in weight-matrix: Dimension ", weights[layer_num].shape)
                for layer_num in range(self.layer_number - 1):
                    if bias[layer_num].shape == (self.layer_infos[layer_num + 1], 1):
                        self.bias.append(bias[layer_num])
                    else:
                        print("Error in bias-matrix: Dimension ", bias[layer_num].shape)
            else:
                print("Wrong weight or bias format!")
                print(type(bias))
                sys.exit()

    def print_nn_info(self):
        """This method prints information about the layer."""
        print("A CNN with " + str(self.layer_number) + " layers.")
        print("Activation Function used:", self.activation_function)
        print("Initializer used:", self.print_initilizer())
        for layer_num in range(self.layer_number):
            print("Layer", str(layer_num), "with", self.layer_infos[layer_num], "nodes")
        try:
            for i in range(len(self.bias)):
                layer_str = "Weight Matrix. Layer " + \
                    str(i) + " -> " + str(i + 1) + " Matrix " + "\n"
                bias_str = "Layer " + str(i + 1) + " Bias Vector." + "\n"
                print(layer_str, self.weights[i])
                print(bias_str, self.bias[i])
            print("Input Size:", self.layer_infos[0], "x 1")
        except IndexError:
            print("\n\n\nWRONG Weights or Bias importet!")
            sys.exit()

    def activate(self, inp):
        """Activation:

        This functions applies the chosen activation-function on a given input.
        :parameter
            inp:: this is the input, on which the function should be applied on,
             it can be any nx1 dimensional vector.
        """

        if self.activation_function == "sigmoid":
            new = []
            for entry in inp:
                new.append([(1 / (1 + math.exp(-entry.item(0))))])
            return np.matrix(new)
        elif self.activation_function == "relu":
            new = []
            for entry in inp:
                if entry <= 0:
                    new.append([0])
                else:
                    new.append([entry.item(0)])
            return np.matrix(new)
        elif self.activation_function == "step":
            new = []
            for entry in inp:
                if entry <= 0:
                    new.append([0])
                else:
                    new.append([1])
            return np.matrix(new)
        elif self.activation_function == "tanh":
            new = []
            for entry in inp:
                new.append([(2 / (1 + math.exp(-2 * entry.item(0))) - 1)])
            return np.matrix(new)
        elif self.activation_function == "linear":
            new = []
            for entry in inp:
                new.append([(2 / (1 + math.exp(-2 * entry.item(0))) - 1)])
            return np.matrix(new)

    def propagate_forwards(self, inp):
        """Propagation

        This function propagates an input forward through the network.
        It safes the output of each layer in the activation list.
        If activate_all_layers is set to true, every layer is being pumped through the activation-
        function, otherwise this is not done for the last layer.
        :parameter
            inp:: this is the input, which should be propageted forward through the network,
                it should be a nx1 dimensional vector
        Return: The function returns the output from the last layer as a vector.
        """
        self.generate_dropout()
        self.activation = [np.matrix(inp)]

        for layer in range(self.layer_number - 2):
            self.activation.append((self.activate(self.weights[layer]
                                                  * self.activation[-1] + self.bias[layer])))

        if self.activate_all_layers:
            output = self.activate(self.weights[-1] * self.activation[-1] + self.bias[-1])
        else:
            output = self.weights[-1] * self.activation[-1] + self.bias[-1]
        self.activation.append(output)
        return output

    def cost(self):
        """Calculates the Cost"""
        cost = self.target - self.output
        cost = np.linalg.norm(cost)
        cost = np.square(cost)
        return cost

    def test(self, inp):
        return self.propagate_forwards(inp)

    def test_info(self, inp, tar):
        """This functions tests a certain input and gives useful information about the test."""
        self.target = np.matrix(tar)
        self.output = self.propagate_forwards(inp)
        cost = self.cost()
        #print("\nInput: \n", np.matrix(inp), "\nTarget Output: \n",
        # self.target, "\nOutput: \n", self.output, "\nCost: \n", cost)
        print("\nCost: \n", cost)
        return self.output

    def delta(self, layer):
        """This function calculates the delta of a certain layer, used for the gradient vector."""
        if self.activation_function == "sigmoid":
            if layer == self.layer_number - 1:
                if self.activate_all_layers:
                    return np.multiply(np.multiply(self.activation[-1], (1 - self.activation[-1])),
                                       (self.target - self.output))
                else:
                    return self.target - self.output
            else:
                return np.multiply(np.multiply(self.activation[layer], (1 - self.activation[layer])),
                                   np.transpose(self.weights[layer]) * self.delta(layer + 1))
        elif self.activation_function == "relu":
            if layer == self.layer_number - 1:
                if self.activate_all_layers:
                    return np.multiply(self.derivative(self.weights[layer - 1] * self.activation[layer - 1] + self.bias[layer - 1]),
                                       (self.target - self.output))
                else:
                    return self.target - self.output
            else:
                return np.multiply(self.derivative(self.weights[layer - 1] * self.activation[layer - 1] + self.bias[layer - 1]),
                                   np.transpose(self.weights[layer]) * self.delta(layer + 1))



    def derivative(self, inp):
        if self.activation_function == "sigmoid":
            return np.multiply(self.activate(inp), (1 - self.activate(inp)))
        elif self.activation_function == "relu":
            new = []
            for entry in inp:
                if np.maximum(0, entry) == 0:
                    new.append([0])
                else:
                    new.append([1])
            return np.matrix(new)



    def update_b(self, eta, layer):
        """Update Bias:

        This method updates the bias of a certain layer,
        by adding the corresponding gradient multiplied by a learning rate (eta), to the old value.
        :parameter
            eta:: float, resembling the learning rate
            layer:: int, number of the layer in which the bias should be updatet
        """
        for entry in range(len(self.bias[layer])):
            self.bias[layer].itemset(entry, self.bias[layer].item(entry) + eta * self.gradient_v.item(0))
            self.gradient_v = np.delete(self.gradient_v, 0, 0)

    def update_w(self, eta, layer):
        """Update Weights:

         This method updates the weights of a certain layer,
         by adding the corresponding gradient multiplied by a learning rate (eta), to the old value.
         :parameter
            eta:: float, resembling the learning rate
            layer:: int, number of the layer in which the bias should be updatet
        """

        for entry in range(self.weights[layer].shape[0]):
            for target in range(self.weights[layer].shape[1]):
                self.weights[layer].itemset((entry, target), self.weights[layer].item(entry, target)
                                            + eta * self.gradient_v.item(0))
                self.gradient_v = np.delete(self.gradient_v, 0, 0)

    def gradient_w(self, layer, to, fro):
        """Gradient of certain weight.

        This function calculates the gradient of a certain weight, by using the delta and an activation
        :parameter
            layer:: int, layer number to calculate
            to:: int, node-number, to which the weight is connected
            fro:: int, node-number, from which the weight comes
        """
        return self.delta(layer + 1).item(to) * self.activation[layer].item(fro)

    def gradient_b(self, layer, node):
        """Gradient of certain bias.

        This function calculates the gradient of a certain bias, by using the delta-function.
        :parameter
            layer:: int, layer number to calculate
            node:: int, node-number, to which the bias is added
        """
        return self.delta(layer + 1)[node]

    def update(self, eta):
        """Update weights:

        This method updates the parameters of the network with the gradient vector
        and the eta by updating the weights and the bias of each layer. After this the weights which
        have been dropped out will be reactivated for the next step.
        :parameter
            eta:: float, which defines the step size with which the weights should be updated with
        """
        for layer in range(self.layer_number - 1):
            self.update_w(eta, layer)
            self.update_b(eta, layer)

        self.reassign_weights()

    def gradient_vector(self):
        """Gradient Vector:

        This Functions creates the gradient vector, which points in the direction, going
        downhill the most. Through this vector we are trying to reach a good local minimum.
        :return
            np-matrix with size nx1 (data-type of entries: float64)
        """
        vector = []
        for layer in range(self.layer_number - 1):
            for entry in range(self.weights[layer].shape[0]):
                for target in range(self.weights[layer].shape[1]):
                    vector.append([self.gradient_w(layer, entry, target)])
            for entry in range(len(self.bias[layer])):
                vector.append([self.gradient_b(layer, entry)])
        #self.visualizer.flatten_data(self.weights, self.bias) -- not needed right now
        return np.matrix(vector, dtype="Float64")

    def train_batch(self, inp_list, target_list, eta=0.05):
        """Batch Training:

        This methods trains the network with a batch of inputs. It does this by calculating
        the gradient vector for every image in the batch and taking the mean of them all.
        Then it updates the parameters with this mean-gradient-vector.
        :parameter
            inp_list:: list of inputs as np.matrix()
            target_list:: list of targets as np.matrix()
            eta:: float, which defines the step size with which the weights should be updated with
        """
        self.gradient_v = 0
        for i in range(len(inp_list)):
            self.test_info(inp_list[i], target_list[i])
            self.gradient_v += self.gradient_vector()
        self.gradient_v = self.gradient_v / len(inp_list)
        self.update(eta)

    def test_train_single(self, inp, tar, eta=0.05):
        """Training with single point:

        This method trains the network with one given input vector. It then updates
        the weights according to a certain eta (step size).
        :parameter
            inp:: input vector, with which we will train -- np.matrix()
            tar:: target vector, resembling the target output from the net -- np.matrix()
            eta:: float, which defines the step size with which the weights should be updated with
        """
        self.test_info(inp, tar)
        self.gradient_v = self.gradient_vector()
        self.update(eta)

    def generate_dropout(self):
        """call this function for every new batch"""
        self.apply_dropout()
    
    def apply_dropout(self):
        """Apply Dropout
        
        Method to apply dropout based on probability p. Remember that dropout should be applied epoch wise.
        New batch/epoch needs new dropout function. Call self.generate_dropout()"""
        self.weights_old = self.weights
        self.dropout_matrices = []

        p = 1 - self.dropout

        for i in range(len(self.weights)):
            # create a matrix which drops out single units of the network
            self.dropout_matrices.append(np.resize(np.random.binomial(1, p, size=(1, self.weights[i].shape[1])), self.weights[i].shape))
            self.weights[i] = np.multiply(self.weights[i], self.dropout_matrices[i])

    def reassign_weights(self):
        """Reassign weights after backprop:

        This method sets the waits correctly after dropout and backprop."""
        weights_after_backprop = self.weights
        self.weights = []

        inverse_matrices = []
        for matrix in self.dropout_matrices:
            mat_new = []
            for row in matrix:
                row_new = []
                for element in row:
                    if element == 0:
                        row_new.append(1)
                    else:
                        row_new.append(0)
                mat_new.append(row_new)
            inverse_matrices.append(np.matrix(mat_new))

        for weight_num in range(len(weights_after_backprop)):
            y = np.multiply(weights_after_backprop[weight_num], self.dropout_matrices[weight_num]) + \
                np.multiply(self.weights_old[weight_num], inverse_matrices[weight_num])
            self.weights.append(y)

    def save_params(self, title):
        """Save network:

        This method saves the parameters of a network, to then later use a
        pretrained network, by using the read_params function.
        :parameter
            title:: this is a part of the title for the text-document, in which we will save the params
        """
        time_stamp = str(datetime.datetime.now()).replace(' ', '_').replace(':', '')[:-7]
        title = title + '_' + time_stamp + '.txt'
        with open(title, 'w') as f:
            f.write(str(self.layer_infos) + '\n')
            for weight in self.weights:
                f.write(str(weight.tolist()) + '\n')
            for bias in self.bias:
                f.write(str(bias.tolist()) + '\n')
