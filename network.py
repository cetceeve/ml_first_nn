import csv
import time
import numpy as np
from random import shuffle

from inputlayer import InputLayer
from hiddenlayer import HiddenLayer
from outputlayer import OutputLayer

class Network:
    numOfInputs = 10
    numOfHiddens = 6
    numOfOutputs = 4
    bias = -1
    lrate = 0.01

    numOfSuccess = 0
    numOfFailure = 0
    successRateLastTurn = 0

    continueTraining = True
    precision = 0.000002

    def __init__(self):
        # create all layers
        self.inputLayer = InputLayer(self.numOfInputs)
        self.hiddenLayer = HiddenLayer(self.numOfInputs, self.numOfHiddens)
        self.outputLayer = OutputLayer(self.numOfHiddens, self.numOfOutputs)

        types, groundTruths, dataVectors = self.getData()
        t0 = time.time()
        self.trainNetwork(types, groundTruths, dataVectors)
        t1 = time.time()
        print("\nTime: " + str(t1 - t0))

    def getData(self):
        rawData = self.readCSV()
        shuffle(rawData)
        types, groundTruths = self.getGTs(rawData)
        dataVectors = np.array([self._assignBias(row) for row in rawData], float)
        return types, groundTruths, dataVectors

    def _assignBias(self, vector):
        vector[-1] = self.bias
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
        epoCounter = 0
        while self.continueTraining:
            epoCounter += 1
            self.trainEpoch(types, groundTruths, dataVectors, str(epoCounter))

    def trainEpoch(self, types, groundTruths, dataVectors, epoCounter):
        counter = 0
        precRates = []
        for vector, groundTruth in zip(dataVectors, groundTruths):
            self.feedSample(vector, groundTruth)

            # control operation
            counter += 1
            if self.numOfSuccess != 0:
                precRates.append(abs(self.successRateLastTurn - self.numOfFailure/self.numOfSuccess))
                self.successRateLastTurn = self.numOfFailure/self.numOfSuccess
            if counter % 100 == 0:
                print("Epo: " + epoCounter + " Data: " + str(counter) + "/40000 Prec: " + str(1 - self.successRateLastTurn), end="\r")
            if counter % 1000 == 0:
                if sum(precRates)/len(precRates) < self.precision:
                    self.continueTraining = False
                    return
                precRates = []

    def feedSample(self, dataVector, groundTruth):
        actVectorInput = self.inputLayer.feedSample(dataVector)
        actVectorHidden = self.hiddenLayer.feedSample(actVectorInput)
        actVectorOutput = self.outputLayer.feedSample(actVectorHidden)

        errorVector = self.getErrorVector(actVectorOutput, groundTruth)
        self.predict(actVectorOutput, groundTruth)
        self.backprop(errorVector, actVectorHidden, actVectorInput)

    def getErrorVector(self, actVectorOutput, groundTruth):
        return [truth - act for act, truth in zip(actVectorOutput, groundTruth)]

    def backprop(self, errorVector, actVectorHidden, actVectorInput):
        self.outputLayer.backprop(self.lrate, errorVector, actVectorHidden)
        self.hiddenLayer.backprop(self.lrate, actVectorInput, errorVector, self.outputLayer)

    def predict(self, actVectorOutput, groundTruth):
        if groundTruth[np.argmax(actVectorOutput)] > 0:
            self.numOfSuccess += 1
        else:
            self.numOfFailure += 1

if __name__ == "__main__":
    Network()
