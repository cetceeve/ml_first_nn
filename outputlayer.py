import numpy as np
from neuron import Neuron

class OutputLayer():
    def __init__(self, preLayer, currentLayer):
        self.weights = np.random.rand(currentLayer, preLayer)
        self.neurons = [Neuron() for i in range(currentLayer)]