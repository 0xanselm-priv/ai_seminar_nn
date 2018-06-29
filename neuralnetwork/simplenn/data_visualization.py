import numpy as np


class DataVis():
    
    def __init__(self):
        self.flatten_paramters = []
        self.i = 0
        
    def flatten_data(self, weights, biases):
        temp = []
        for i in weights:
            i = i.flatten()
            temp = np.concatenate((temp,i), axis = 0)
        temp = temp.tolist()
        for i in biases:
            i = np.asarray(i)
            i = i.flatten()
            i = i.tolist()
            
            temp.extend(i)
            
        self.i += 1
        self.flatten_paramters.append(temp)
        
        if self.i >= 9998: # <---- adjust here for the number of training samples. look to main.py in line 144
            
            with open("parameters.txt", "w") as f:
                for entry in self.flatten_paramters:
                    f.write("%s\n" % entry)
