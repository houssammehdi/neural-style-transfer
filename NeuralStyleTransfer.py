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
# Checking if cuda (gpu) is available
print(torch.cuda.is_available())

imsize = 512 if torch.cuda.is_available() else 128 
# using bigger image size if GPU available, if not we use a small size for the CPU

# Loading the style and content images
style_img = image_loader("images/1920x1200.jpg")
content_img = image_loader("images/nature.jpg")


# Resizing and converting image to tensor
loader = transforms.Compose([
    transforms.Resize(imsize), 
    transforms.ToTensor()])


# Reconverting from tensor to image
unloader = transforms.ToPILImage()

 
# Displays an image by reconverting a copy of it to PIL 
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

        
# Content Loss Module
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

    
# Gram Matrix computation
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
    
    
# Style Loss Module
class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        # same as the Content Loss, but we used the gram matrix

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        # mean square error between GXL and GSL ||GXL-GSL||^2
        return input
    
    
# importing pre-trained model, evaluation mode   
# normalizing the images before sending them to the network
cnn = models.vgg19(pretrained=True).features.to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)


# Creating a module to normalize input image so we can easily put it in a nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std
    
    
# desired depth layers to compute style and content losses
# the layers can be changed but we will use these layers as
# they have been proven to give great results for the topic
content_layers = ['conv_4']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


# Computing the style and content losses
def get_style_and_losses(style_img, content_img):

    normalization = Normalization(cnn_normalization_mean, cnn_normalization_std).to(device)
    # Getting the Normalization module
    content_losses = []
    style_losses = []
    # Iterable access to content and style losses
    model = nn.Sequential(normalization)
    # New nn.Sequential -> Normalization sequencial activation

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
            # Looping over Conv2d layer and renaming to conv_(layer number)
        
        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            # Getting the content loss from the module
            model.add_module("content_loss_{}".format(i), content_loss)
            # Adding the module to the model
            content_losses.append(content_loss)
            # Adding the content loss to the Content_losses List 

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            # Getting the style loss from the module
            model.add_module("style_loss_{}".format(i), style_loss)
            # Adding the module to the model
            style_losses.append(style_loss) 
            # Adding the style loss to the Style_losses List

    # Removing the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            # If the layer is an instance of ContentLoss or StyleLoss
            break # Stop the loop ( stop incrementing i )

    model = model[:(i + 1)] # Everything until i + 1 -> 0 to i

    return model, style_losses, content_losses


# Using a noise image as an input image which is generated from the content image
input_img = torch.randn(content_img.data.size(), device=device)


# Optimzer L-BFGS
def get_input_optimizer(input_img):
    # We will use L-BFGS algorithm to run our gradient descent
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    # Creating a PyTorch L-BFGS optimizer and we pass the image as the tensor to optimize 
    return optimizer


