from inputlayer import InputLayer
from hiddenlayer import HiddenLayer
from outputlayer import OutputLayer

class Network:
    numOfInputs = 10
    numOfHiddens = 12
    numOfOutputs = 4

    dataVector = (0.772040289,0.512995131,-1.261525632,-1.10916658,1.105782323,-1.1127501,0.516377072,0.270915513,0.410573473, -1)

    def __init__(self):
        # create all layers
        self.inputLayer = InputLayer(self.numOfInputs)
        self.hiddenLayer = HiddenLayer(self.numOfInputs, self.numOfHiddens)
        self.outputLayer = OutputLayer(self.numOfHiddens, self.numOfOutputs)

        self.feedSample(self.dataVector)
    
    def feedSample(self, dataVector):
        actInput = self.inputLayer.feedSample(dataVector)
        print(actInput)
        actHidden = self.hiddenLayer.feedSample(actInput)
        print(actHidden)
        actOutput = self.outputLayer.feedSample(actHidden)
        print(actOutput)
    
    def backpropagate(self):
        pass

if __name__ == "__main__":
    Network()