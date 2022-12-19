import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, models, transforms
import torchvision
import torchmetrics as tm
import csv
import pandas as pd

def savestatis(trainloss, validloss, trainaccu, validaccu):
    with open("/home/meng/Yiping/model/output_n32.csv", 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(trainloss)
        w.writerow(validloss)
        w.writerow(trainaccu)
        w.writerow(validaccu)

def load_(root):
    Data = pd.read_csv(root, delimiter= ',', encoding= 'utf-8', header= None)
    data = Data.to_numpy()
    trainloss = []
    validloss = []
    trainaccu = []
    validaccu = []

    for i in range(0, np.size(data, axis= 1)):
        trainloss.append(data[0][i])
        validloss.append(data[1][i])
        trainaccu.append(data[2][i])
        validaccu.append(data[3][i])
    
    return trainloss, validloss, trainaccu, validaccu

def loadmodel(root, model, opt, device):
    checkpoint = torch.load(root, map_location= device)
    model_state, optimizer_state = checkpoint["model"], checkpoint["optimizer"]
    model.load_state_dict(model_state)
    opt.load_state_dict(optimizer_state)
    return model, opt

def accuracy(output, tar):
    pred = []
    for i in range(output.size()[0]):
        if output[i][0] > output[i][1]:
            pred.append(0)
        else: pred.append(1)

    pred = torch.tensor(pred, dtype= float, device= "cuda")
    acc = (pred == tar).sum().item() / pred.size()[0]

    return acc

