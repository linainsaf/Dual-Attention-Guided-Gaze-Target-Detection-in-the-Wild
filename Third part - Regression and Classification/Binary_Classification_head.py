#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 21:43:34 2022

@author: delphinedoutsas
"""

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import tensorflow as tf
import torch
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import torchvision.models as models

#device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')


#######################################################################
#                                                                     #
#           Definition of the functions for the project               #
#                                                                     #
#######################################################################

def concat(p1, p2, axis=0):
    """
        p1, p2 : already tensors
        axis   : 0 for raw, 1 for columns
        
        Concatenation of the two pictures of the entry
    """
    return tf.concat([p1, p2], 0)



#######################################################################
#                                                                     #
#             Proceeding of the extraction of the data                #
#                                                                     #
#######################################################################

preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])
    

#######################################################################
#                                                                     #
#                              ResNet50                               #
#                                                                     #
#######################################################################  

backbone = models.resnet50(pretrained=True, progress=True)
print("ResNet50 loaded")
# Freeze the learning of the layers of the resnet50
for param in backbone.parameters():
    param.requires_grad = False
    
    
    
#######################################################################
#                                                                     #
#           Création du modèle "Binary Classification Head"           #
#                                                                     #
#######################################################################   
    
    
class binary(nn.Module):
    def __init__(self):
        super(binary, self).__init__()
        
        #Resnet
        self.backbone = backbone
        
        #convolution layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding="same")
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding="same")

        # Fully connected layer
        self.fc1 = nn.Linear(in_features=1000, out_features=2)
        
    # x represents our data
    def forward(self, x):
        x = self.backbone(x)
        
        x = x.unsqueeze(dim=1) #unsqueeze by adding the second dimension (in_channels=1)
        x = self.conv1(x)

        x = self.conv2(x)
        x = x.squeeze(dim=1)
        x = self.fc1(x)
        
        output = F.log_softmax(x, dim=1)
        return output


#  Only a simple test on the Binary Classification Head  
#--------------------------------------------------------

# Equates to one random 448x224 image
random_data = torch.rand((1, 3, 224+224, 224))

my_nn = binary()
result = my_nn(random_data)
print(result)






