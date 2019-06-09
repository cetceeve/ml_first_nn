import numpy as np

class Neuron:
    def __init__(self):
        self.act = 0
        self.firstDeriv = 0

    def feedForward(self, inputs, weights):
        scalar = np.dot(inputs, weights)
        self.act = np.tanh(scalar)
        self.firstDeriv = 1.0 - np.tanh(scalar)**2
        return self.act

    def directActivation(self, _input):
        self.act = np.tanh(_input)
        self.firstDeriv = 1.0 - np.tanh(_input)**2
        return self.act
