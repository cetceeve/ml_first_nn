import csv
import numpy as np

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
        # types, groundTruths, dataVectors = self.getData()
        
    def getData(self):
        rawData = self.readCSV()
        types, groundTruths = self.getGTs(rawData)
        dataVectors = np.array([row[:-1] for row in rawData], float)
        return types, groundTruths, dataVectors
        
    def readCSV(self):
        with open("samples_4_classes_normalized.csv", mode="r") as dataFile:
            return list(csv.reader(dataFile))[1:]
    
    def getGTs(self, rawData):
        types = list({row[len(row) - 1] for row in rawData})
        groundTruths = np.full((len(rawData), len(types)), -1, int)
        
        for i, row in enumerate(rawData):
            groundTruths[i][types.index(row[len(row) - 1])] = 1
        return types, groundTruths

    def feedSample(self, dataVector):
        actInput = self.inputLayer.feedSample(dataVector)
        actHidden = self.hiddenLayer.feedSample(actInput)
        actOutput = self.outputLayer.feedSample(actHidden)
    
    def backpropagate(self):
        pass

if __name__ == "__main__":
    Network()