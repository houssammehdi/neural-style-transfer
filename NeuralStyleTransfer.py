import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models
from google.colab import files

import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

imsize = 512 if torch.cuda.is_available() else 128 

loader = transforms.Compose([
    transforms.Resize(imsize), 
    transforms.ToTensor()])

unloader = transforms.ToPILImage()
   
def imshow(tensor, title=None):
    image = tensor.cpu().clone()  
    image = image.squeeze(0)   
    image = unloader(image)
    image.save('output.jpg')
    files.download('output.jpg')
    plt.imshow(image)
    if title is not None:
        plt.title(title)

        
class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    a, b, c, d = input.size()  
    # a=batch size
    # b=number of feature maps
    # c,d=dimensions of a feature map
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())  
    return G.div(a * b * c * d)
