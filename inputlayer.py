import numpy as np
from neuron import Neuron

class InputLayer():
    def __init__(self, numOfNeurons):
        # self.weights = np.random.rand(currentLayer, preLayer)
        self.neurons = [Neuron() for i in range(numOfNeurons)]