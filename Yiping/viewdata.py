import torch
import torchvision
import torchvision.transforms as transforms

device = "cuda"
trans = transforms.Compose([
              transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

trans_resize = transforms.Compose([
              transforms.Resize([300, 300], interpolation= 2),
              transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

ori = torchvision.datasets.PCAM(root="YiPing", split = 'train', download= False, transform= trans)
res = torchvision.datasets.PCAM(root="YiPing", split = 'train', download= False, transform= trans_resize)

batch_size = 1
origin = torch.utils.data.DataLoader(dataset = ori, batch_size = batch_size, shuffle = False)
resized = torch.utils.data.DataLoader(dataset = res, batch_size = batch_size, shuffle = False)

import matplotlib.pyplot as plt
import PIL
from PIL import Image
def plot(data, predict, n):
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.gray()
    plt.imshow(data)
    plt.title("Input")

    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.gray()
    plt.imshow(predict)
    plt.title(f"Output of first {n} layers")

    plt.show()

def plotimg(img, img2):
    plt.subplot(2, 4, 1)
    plt.axis("off")
    plt.gray()
    plt.imshow(img[0])
    plt.title("Channel R")

    plt.subplot(2, 4, 2)
    plt.axis("off")
    plt.gray()
    plt.imshow(img[1])
    plt.title("Channel G")

    plt.subplot(2, 4, 3)
    plt.axis("off")
    plt.gray()
    plt.imshow(img[2])
    plt.title("Channel B")
  
    img = img.numpy().transpose((1, 2, 0))
    plt.subplot(2, 4, 4)
    plt.axis("off")
    plt.imshow(img)
    plt.title("Origin Image")
    
    plt.subplot(2, 4, 5)
    plt.axis("off")
    plt.gray()
    plt.imshow(img2[0])
    plt.title("Channel R")

    plt.subplot(2, 4, 6)
    plt.axis("off")
    plt.gray()
    plt.imshow(img2[1])
    plt.title("Channel G")

    plt.subplot(2, 4, 7)
    plt.axis("off")
    plt.gray()
    plt.imshow(img2[2])
    plt.title("Channel B")

    img2 = img2.numpy().transpose((1, 2, 0))
    plt.subplot(2, 4, 8)
    plt.axis("off")
    plt.imshow(img2)
    plt.title("Resized Image")

    plt.show()

ito = iter(origin)
itr = iter(resized)
for i in range(len(origin)):
    img, lab = next(ito)
    img2, lab2 = next(itr)

    plotimg(img[0], img2[0])

