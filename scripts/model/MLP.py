#!/usr/bin/python3

import torch
import torch.nn as nn
import json
from datetime import datetime

def getAccuracy(model):
    model.eval()
    correct = 0.0
    for (i,x) in enumerate(testX):
        y = model(x)
        if (testY[i] == 0 and y < 0.5) or (testY[i] == 1 and y >= 0.5):
            correct += 1.0
    model.train()
    return correct / len(testY)

def saveModel(model, path, epoch):
    file = path + "/model_" + str(epoch) + ".pth"
    torch.save(model.state_dict(), file)


def loadModel(file: str, modelNumber: int):
    model = None
    if modelNumber == 1:
        model = nn.Sequential(nn.Linear(512, 256), nn.Linear(256, 1), nn.Sigmoid())
    if modelNumber == 2:
        model = nn.Sequential(nn.Linear(512, 256), nn.Linear(256, 64), nn.Linear(64, 1), nn.Sigmoid())
    if modelNumber == 3:
        model = nn.Sequential(nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 64), nn.Linear(64, 1), nn.Sigmoid())
    if modelNumber == 4:
        model = nn.Sequential(nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 64), nn.Linear(64, 32), nn.Linear(32, 1), nn.Sigmoid())
    model.load_state_dict(torch.load(file))
    model.eval()
    return model


if __name__ == "__main__":
    print("Model number (1-4): ", end="")
    modelNumber = int(input())

    print("Save folder path: ", end="")
    saveFolder = input()

    epochs = 4000

    if modelNumber == 1:
        model = nn.Sequential(nn.Linear(512, 256), nn.Linear(256, 1), nn.Sigmoid())
    if modelNumber == 2:
        model = nn.Sequential(nn.Linear(512, 256), nn.Linear(256, 64), nn.Linear(64, 1), nn.Sigmoid())
    if modelNumber == 3:
        model = nn.Sequential(nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 64), nn.Linear(64, 1), nn.Sigmoid())
    if modelNumber == 4:
        model = nn.Sequential(nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 64), nn.Linear(64, 32), nn.Linear(32, 1), nn.Sigmoid())




    print("loading dataset")

    rawData = []
    with open("../datasets/learningDataset.json", "r") as f:
        rawData = json.load(f)

    trainSamples = int(len(rawData) * 0.8)

    rawTrainData = rawData[:trainSamples] 
    rawTestData = rawData[trainSamples:]


    trainX = torch.tensor([x["d"] for x in rawTrainData])
    testX = torch.tensor([x["d"] for x in rawTestData])


    trainY = torch.tensor([[float(x["c"])] for x in rawTrainData])
    testY = torch.tensor([[float(x["c"])] for x in rawTestData])

    print(sum(float(x["c"]) for x in rawTestData))
    print(len(rawTestData))


    print("training started")

    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.02)

    for e in range(epochs):
        start = datetime.now()
        predictions = model(trainX)
        loss = criterion(predictions, trainY)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        testPredictions = model(testX)
        testLoss = criterion(testPredictions, testY)

        print("epoch: " + str(e) + ", training loss: " + str(loss.item()) + ", testing loss: " + str(testLoss.item()) + ", time: " + str((datetime.now() - start).total_seconds()) + " s")
        if e % 10 == 0:
            print("------------------------------------------")
            print("Epoch " + str(e) + " test accuracy: " + str(getAccuracy(model)))
            print("------------------------------------------")
            saveModel(model, saveFolder, e)

