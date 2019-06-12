import numpy as np
from neuron import Neuron


class HiddenLayer():
    def __init__(self, numNeuronsPreLayer, numNeuronsCurrentLayer, func):
        self.weights = np.random.uniform(low=0.0, high=0.1, size=(numNeuronsCurrentLayer, numNeuronsPreLayer))
        self.neurons = [Neuron(func) for i in range(numNeuronsCurrentLayer)]

    def feedSample(self, dataVector):
        return [neuron.feedForward(dataVector, self.weights[i]) for i, neuron in enumerate(self.neurons)]

    def backprop(self, lrate, actVectorInput, errorVector, outputLayer):
        # neue gewichte = bisherige gewichte * lrate * aktivierung input * ableitung aktivierungsfunktions hier * summe über output neuronen((GT - act output) * first deriv output * weight)
        for j, neuron in enumerate(self.neurons):
            for i, actInput in enumerate(actVectorInput):
                self.weights[j][i] += lrate * actInput * neuron.firstDeriv * self._sumErrorOutputLayer(errorVector, outputLayer, j)

    def _sumErrorOutputLayer(self, errorVector, outputLayer, hiddenLayerNeuronIndex):
        return sum([errorVector[k] * neuron.firstDeriv * outputLayer.weights[k][hiddenLayerNeuronIndex] for k, neuron in enumerate(outputLayer.neurons)])
