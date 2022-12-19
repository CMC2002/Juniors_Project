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

device = "cuda"
model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', weights= 'DEFAULT')

classifier = nn.Sequential(
        nn.Linear(model.fc.in_features, 2)
        )
model.fc = classifier

model = model.to(device)
opt = optim.AdamW(model.parameters(), lr= 0.001)
model, _= func.loadmodel("/home/meng/Yiping/model/checkpoint_n32.ckpt", model, opt, device= device)


trans_v = transforms.Compose([transforms.Resize([300, 300], interpolation=2),
              transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
validset = torchvision.datasets.PCAM(root="YiPing", split = 'val', download= True, transform= trans_v)
loader = torch.utils.data.DataLoader(dataset = validset, batch_size = 1, shuffle = False)

model.eval()
outputs = []
GT = []
accuracy = 0

from tqdm import tqdm
for data, target in tqdm(loader):
    with torch.no_grad():
        output = model(data.to(device))
        _, predicted = torch.max(output.data, 1)
        
        GT.append(target.cpu().numpy()[0])
        accuracy += (predicted == target.to(device)).sum().item()
        outputs.append(predicted.cpu().numpy()[0])
accuracy /= len(loader)

import matplotlib as plt
GT = np.array(GT, dtype= int)
outputs = np.array(outputs, dtype= int)

from numpy import save
save("groundtruth.npy", GT)
save("outputs.npy", outputs)

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
print("AUC= ", round(roc_auc_score(outputs, GT),3))
fpr, tpr, thresholds = roc_curve(outputs, GT, pos_label= 1)
plt.plot(fpr, tpr, color= 'red', label= 'ROC curve (area = %0.2f)')
plt.grid(color= 'grey', linewidth= 0.5)
plt.plot([0,1], [0,1], color= 'green', linestyle= '--')
plt.show()

from sklearn.metrics import accuracy_score
print("Accuracy= ", round(accuracy_score(outputs, GT), 3))

from sklearn.metrics import confusion_matrix
cnf = confusion_matrix(outputs, GT)
tp = cnf[0][0]
fp = cnf[0][1]
tn = cnf[1][1]
fn = cnf[1][0]
print("Sensitivity= ", round(tp/(tp+fn), 3))
print("Specificity= ", round(tn/(tn+fp), 3))
print("Precision= ", round(tp/(tp+fp), 3))
print("Negative Predictive Value= ", round(tn/(tn+fn), 3))
