from inputlayer import InputLayer
from hiddenlayer import HiddenLayer
from outputlayer import OutputLayer

class Network:
    numOfInputs = 9
    numOfHiddens = 12
    numOfOutputs = 4

    def __init__(self):
        # create all layers
        self.inputLayer = InputLayer(self.numOfInputs)
        self.hiddenlayer = HiddenLayer(self.numOfInputs, self.numOfHiddens)
        self.outputlayer = OutputLayer(self.numOfHiddens, self.numOfOutputs)

    def feedForward(self):
        pass
    
    def backpropagate(self):
        pass