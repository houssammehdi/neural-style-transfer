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
# resizing and converting image to tensor

unloader = transforms.ToPILImage()
# reconverting from tensor to image
   
def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # clone tensor 
    image = image.squeeze(0)  # remove first dimension
    image = unloader(image) # reconvert to image
    image.save('output.jpg')
    files.download('output.jpg')
    # download image (used for colab)
    plt.imshow(image)
    if title is not None:
        plt.title(title)

        
class ContentLoss(nn.Module):

    def __init__(self, target,):
        # constructor takes feature maps as input -> returns the weighted content distance
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        # dynamically computing the gradients

    def forward(self, input):
        # transparent content loss layer
        self.loss = F.mse_loss(input, self.target)
        # ∥FXL−FCL∥ ^ 2
        # loss saved as a parameter of the module
        return input

def gram_matrix(input):
    a, b, c, d = input.size()  
    # a=batch size
    # b=number of feature maps
    # c,d=dimensions of a feature map
    features = input.view(a * b, c * d)
    # reshaping FXL into ^FXL -> KxN matrix 
    # K = number of feature maps at layer L
    # N = lenght of vectorized feature map
    G = torch.mm(features, features.t())  
    # multiplying matrix by transpose

    return G.div(a * b * c * d)
    # normalizing by dividing by the number of elements in each f. map
    # crucial step because larger values have larger impact during gradient descent
