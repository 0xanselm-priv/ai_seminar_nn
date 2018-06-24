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


class Main():

    def __init__(self):
        self.main()

    def main(self):
        a = network_class.Network([(2), (3), (2)], weights=[np.matrix([[0.4, 2], [-1.3, 0.1], [0.4, 0.5]]), np.matrix([[0.6, 0.2, -0.9], [0.4, 1.1, 0.7]])], bias=[np.matrix([[0], [0.3], [2]]), np.matrix([[0.6], [0.5]])],activation_function="sigmoid", initilizer="predefined")
        a.test_train([[2], [1]], [[1], [0]])

        
if __name__ == "__main__":
    Main()

