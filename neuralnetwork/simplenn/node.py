import math

class Node(object):
    def __init__(self, bias, number):
        self.weight = []
        self.bias = bias
        self.name = number

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def activate(self):
        print("activated")

    def read_output(self, output):
        return output

    def add_weight(self, w):
        self.weight.append(w)