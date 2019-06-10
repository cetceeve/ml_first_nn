import csv
import time
import numpy as np

from inputlayer import InputLayer
from hiddenlayer import HiddenLayer
from outputlayer import OutputLayer

class Network:
    numOfInputs = 10
    numOfHiddens = 12
    numOfOutputs = 4

    def __init__(self):
        # create all layers
        self.inputLayer = InputLayer(self.numOfInputs)
        self.hiddenLayer = HiddenLayer(self.numOfInputs, self.numOfHiddens)
        self.outputLayer = OutputLayer(self.numOfHiddens, self.numOfOutputs)

        types, groundTruths, dataVectors = self.getData()
        self.trainNetwork(types, groundTruths, dataVectors)
        
    def getData(self):
        rawData = self.readCSV()
        types, groundTruths = self.getGTs(rawData)
        dataVectors = np.array([self._assignBias(row) for row in rawData], float)
        return types, groundTruths, dataVectors

    def _assignBias(self, vector):
        vector[-1] = -1
        return vector

    def readCSV(self):
        with open("samples_4_classes_normalized.csv", mode="r") as dataFile:
            return list(csv.reader(dataFile))[1:]
    
    def getGTs(self, rawData):
        types = list({row[-1] for row in rawData})
        groundTruths = np.full((len(rawData), len(types)), -1, int)
        
        for i, row in enumerate(rawData):
            groundTruths[i][types.index(row[-1])] = 1
        return types, groundTruths


    def trainNetwork(self, types, groundTruths, dataVectors):
        t0 = time.time()
        self.trainEpoch(types, groundTruths, dataVectors)
        t1 = time.time()
        print(t1 - t0)

    def trainEpoch(self, types, groundTruths, dataVectors):
        counter = 0
        for vector, groundTruth in zip(dataVectors, groundTruths):
            self.feedSample(vector, groundTruth)
            counter += 1
            print(str(counter) + "/40000", end="\r")
            # break # open up loop when ready

    def feedSample(self, dataVector, groundTruth):
        actInput = self.inputLayer.feedSample(dataVector)
        actHidden = self.hiddenLayer.feedSample(actInput)
        actOutput = self.outputLayer.feedSample(actHidden)
    
    def backpropagate(self):
        pass

if __name__ == "__main__":
    Network()