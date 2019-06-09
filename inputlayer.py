import numpy as np
from neuron import Neuron

class InputLayer():
    def __init__(self, numOfNeurons):
        self.weights = np.random.rand(numOfNeurons, numOfNeurons)
        self.neurons = [Neuron() for i in range(numOfNeurons)]

    def feedSample(self, dataVector):
        return [neuron.feedForward(dataVector, self.weights[i]) for i, neuron in enumerate(self.neurons)]