import numpy as np


class Neuron:
    def __init__(self, func=None):
        self.act = 1.
        self.firstDeriv = 1.
        self.func = func

    def feedForward(self, inputs, weights):
        scalar = np.dot(inputs, weights)
        if self.func == "tanh":
            self._tanh(scalar)
        elif self.func == "relu":
            self._relu(scalar)
        elif self.func == "sig":
            self._sig(scalar)
        # elif self.func == "softmax":
            # self._softmax(scalar)
        return self.act

    def directActivation(self, x):
        self.act = x

        # sigmoid
        # if x < 0:
        #     x = 0
        # self.act = 1 / (1 + np.exp(-x))
        # self.act = np.tanh(x)
        return self.act

    def _tanh(self, x):
        self.act = np.tanh(x)
        self.firstDeriv = 1.0 - np.tanh(x)**2

    def _relu(self, x):
        self.act = max(0.0, x)
        if x < 0:
            self.firstDeriv = 0.0
        else:
            self.firstDeriv = 1.0

    def _sig(self, x):
        self.act = 1 / (1 + np.exp(-x))
        self.deriv = self.act * (1 - self.act)

    # def _softmax(self, x):
    #     self.act = np.exp(x) / np.sum(np.exp(x))
