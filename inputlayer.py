import numpy as np
from neuron import Neuron


class InputLayer():
    def __init__(self, numOfNeurons):
        # self.weights = np.ones((numOfNeurons, numOfNeurons))
        self.neurons = [Neuron() for i in range(numOfNeurons)]

    def feedSample(self, dataVector):
        # return [neuron.feedForward(dataVector, self.weights[i]) for i, neuron in enumerate(self.neurons)]
        return [neuron.directActivation(dataVector[i]) for i, neuron in enumerate(self.neurons)]
