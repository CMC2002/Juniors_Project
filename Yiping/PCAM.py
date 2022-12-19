import Functions as func
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
import model as m

device = "cuda" if torch.cuda.is_available() else "cpu"

trans = transforms.Compose([transforms.RandomVerticalFlip(),
              transforms.RandomHorizontalFlip(),
              transforms.RandomRotation(30, center=(48, 48)),
              transforms.Resize([300, 300], interpolation=2),
              transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
trans_v = transforms.Compose([transforms.Resize([300, 300], interpolation=2),
              transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
trainset = torchvision.datasets.PCAM(root="YiPing", split = 'train', download= True, transform= trans)
validset = torchvision.datasets.PCAM(root="YiPing", split = 'val', download= True, transform= trans_v)

batch_size = 32
train_loader = torch.utils.data.DataLoader(dataset = trainset, batch_size = batch_size, shuffle = True)
valid_loader = torch.utils.data.DataLoader(dataset = validset, batch_size = batch_size, shuffle = False)

myseed = 998244353  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

trainaccu = []
validaccu = []
trainloss = []
validloss = []

'''
from torchsummary import summary
model = inceptionV3()
model = model.to(device)
summary = summary(model, (3, 512, 512))
print(summary)
'''

from tqdm import tqdm

num_iter= 100
model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', weights= 'DEFAULT')

classifier = nn.Sequential(
        nn.Linear(model.fc.in_features, 2)
        )
model.fc = classifier

model = model.to(device)
criterion = nn.CrossEntropyLoss()
best_acc = 0
patience = 8
stale = 0
learning_rate = 0.00001
opt = optim.AdamW(model.parameters(), lr= learning_rate)
## model, opt = func.loadmodel("/home/meng/Yiping/model/checkpoint.ckpt", model, opt, device= device)
## trainloss, validloss, trainaccu, validaccu = func.load_("/home/meng/Yiping/model/output.csv")

for epoch in range(0, num_iter):
    print("Epoch", epoch)

    model.train()

    train_loss = []
    train_accs = []
    for batch in tqdm(train_loader):
        imgs, labels = batch

        output, _ = model(imgs.to(device))
        loss = criterion(output, labels.to(device))

        opt.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm = 10)
        opt.step()

        _, predicted = torch.max(output.data, 1)
        acc = (predicted == labels.to(device)).sum().item() / batch_size
        train_loss.append(loss.item())
        train_accs.append(acc)

    train_loss = sum(train_loss) / len(train_loss)
    train_accs = sum(train_accs) / len(train_accs)

    trainloss.append(train_loss)
    trainaccu.append(train_accs)
    print(f"[ Train | {epoch + 1:03d}/{num_iter:03d} ] loss = {train_loss:.5f}, acc = {train_accs:.5f}")

    model.eval()

    valid_loss = []
    valid_accs = []

    for batch in tqdm(valid_loader):

        imgs, labels = batch

        with torch.no_grad():
            output = model(imgs.to(device))
            loss = criterion(output, labels.to(device))
            _, predicted = torch.max(output.data, 1)

            acc = (predicted == labels.to(device)).sum().item() / batch_size
            valid_loss.append(loss.item())
            valid_accs.append(acc)
    
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_accs = sum(valid_accs) / len(valid_accs)

    print(f"[ Valid | {epoch + 1:03d}/{num_iter:03d} ] loss = {valid_loss:.5f}, acc = {valid_accs:.5f}")

    validloss.append(valid_loss)
    validaccu.append(valid_accs)

    func.savestatis(trainloss, validloss, trainaccu, validaccu)
    if valid_accs > best_acc:
        with open(f"/home/meng/Yiping/model/log_n32.txt", "a") as f:
            f.write(f"[ Valid | {epoch + 1:03d}/{num_iter:03d} ] loss = {valid_loss:.5f}, acc = {valid_accs:.5f} -> best\n")
        
        print(f"Best model found at epoch {epoch}, saving model")
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict()}, "/home/meng/Yiping/model/checkpoint_n32.ckpt")
        best_acc = valid_accs
        stale = 0
    
    else:
        stale += 1
        if stale >= patience:
            print(f"No improvment {patience} consecutive epochs, early stopping")
            break

