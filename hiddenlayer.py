import numpy as np
from neuron import Neuron

class HiddenLayer():
    def __init__(self, numNeuronsPreLayer, numNeuronsCurrentLayer):
        self.weights = np.random.rand(numNeuronsCurrentLayer, numNeuronsPreLayer)
        self.neurons = [Neuron() for i in range(numNeuronsCurrentLayer)]

    def feedSample(self, dataVector):
        return [neuron.feedForward(dataVector, self.weights[i]) for i, neuron in enumerate(self.neurons)]

    def backprop(self, lrate):
        # neue gewichte = bisherige gewichte * lrate * aktivierung input * ableitung aktivierungsfunktions hier * summe von(error * ableitung aktivierungsfunktion output * aktivierung hidden)
        pass