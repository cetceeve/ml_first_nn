import numpy as np
from neuron import Neuron

class OutputLayer():
    def __init__(self, numNeuronsPreLayer, numNeuronsCurrentLayer):
        self.weights = np.random.rand(numNeuronsCurrentLayer, numNeuronsPreLayer)
        self.neurons = [Neuron() for i in range(numNeuronsCurrentLayer)]

    def feedSample(self, dataVector):
        return [neuron.feedForward(dataVector, self.weights[i]) for i, neuron in enumerate(self.neurons)]

    def backprop(self, lrate, errorVector, actVectorHidden):
        # iteration über output neuronen
        # neue gewichte = bisherige gewichte - lrate * error * first deriv output * act hidden
        for i, neuron in enumerate(self.neurons):
            for j, actHidden in enumerate(actVectorHidden):
                self.weights[i][j] += lrate * errorVector[i] * neuron.firstDeriv * actHidden