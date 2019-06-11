import numpy as np
from neuron import Neuron


class OutputLayer():
    def __init__(self, numNeuronsPreLayer, numNeuronsCurrentLayer, func):
        self.weights = np.random.uniform(low=0.0, high=0.1, size=(numNeuronsCurrentLayer, numNeuronsPreLayer))
        self.neurons = [Neuron(func) for i in range(numNeuronsCurrentLayer)]

    def feedSample(self, dataVector):
        return [neuron.feedForward(dataVector, self.weights[i]) for i, neuron in enumerate(self.neurons)]

    def backprop(self, lrate, errorVector, actVectorHidden):
        # iteration über output neuronen
        # neue gewichte = bisherige gewichte - lrate * error * first deriv output * act hidden
        for i, neuron in enumerate(self.neurons):
            for j, actHidden in enumerate(actVectorHidden):
                self.weights[i][j] += lrate * errorVector[i] * neuron.firstDeriv * actHidden
