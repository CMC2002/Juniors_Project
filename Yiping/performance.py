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
import Functions as func
import torch.optim as optim

model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', weights= 'DEFAULT')

classifier = nn.Sequential(
        nn.Linear(model.fc.in_features, 2)
        )
model.fc = classifier

model = model.to(device)
opt = optim.AdamW(model.parameters(), lr= 0.001)
model, _= func.loadmodel("/home/meng/Yiping/model/checkpoint.ckpt", model, opt, device= device)

loader = torch.utils.data.DataLoader(dataset = validset, batch_size = 1, shuffle = False)

model.eval()
accuracy = []
output = []

from tqdm import tqdm
for data, target in tdqm(loader):
