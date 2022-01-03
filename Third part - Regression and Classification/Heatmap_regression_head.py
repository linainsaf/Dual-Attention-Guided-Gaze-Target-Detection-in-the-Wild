#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 21:15:40 2022

@author: delphinedoutsas
"""

# Import for the project
import numpy as np
from os.path import dirname, join as pjoin
import scipy.io as sio
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, datasets
import tensorflow as tf
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torchvision.models as models
import torchvision.transforms as transforms
from torch import nn
import torch.nn.functional as F


data_path= "/content/drive/MyDrive/MLA_Projet/GazeFollow/data/data_new"


#######################################################################
#                                                                     #
#           Definition of the functions for the project               #
#                                                                     #
#######################################################################


# Function to extract the data contained in the matrices 
def dataset_extractor_test(data_path, matrix_contents_dataset):#, dual_att_list):

    images = []
    gazes = []

    for i in range(len(matrix_contents_dataset['test_gaze'][0])):
        
        ########### Image part ############
        pic_path = matrix_contents_dataset['test_path'][i][0][0]
        pic = cv2.imread(pjoin(data_path, pic_path))
        resized_pic = cv2.resize(pic,(224,224))
        #dual_attention_map = dual_att_list[i]
        #cv2.resize(dual_attention_map, (224,224))
        pic = np.concatenate((resized_pic, resized_pic), 1)
        images.append(pic)

        ########### Gaze part #############
        gaze_i = matrix_contents_dataset['test_gaze'][0][i]
        mean_pts = np.mean(gaze_i, axis=0)
        real_pts = mean_pts*(64, 64)
        #creation of a picture with 0 everywhere and 1 on the gaze point
        target = np.zeros((64, 64, 1))
        target[int(real_pts[0]), int(real_pts[1]), :] = 255
        gazes.append(target)

    return images , gazes


# Function to preprocess the data before passing them through the networks
class GazeFollowDataset(Dataset):

    def __init__(self, data_in, target, transform=None):
        self.data_in            = data_in
        #self.dual_attention_map = dual_attention_map
        self.target             = target
        self.transform          = transform

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.data_in[idx]
        y = self.target[idx]

        if self.transform:
              x = self.transform(x)
              #x.unsqueeze_(0).to(device="cuda")
              y = y.unsqueeze_(0).to(device="cuda")
        
        return x, y
    
    
def concat(p1, p2, axis=0):
    """
        p1, p2 : already tensors
        axis   : 0 for raw, 1 for columns
        
        Crop and resize of the images and concatenation of the two pictures of the entry
    """

    return tf.concat([p1, p2], 0)


def train(model, optimizer, loss_fn, train_dl, val_dl, epochs=100, device='cpu'):
    """
        /!\ Credit to https://inside-machinelearning.com/la-fonction-pytorch-parfaite-pour-entrainer-son-modele/
        
        Function to train the model 
    """
    
    print('train() called: model=%s, opt=%s(lr=%f), epochs=%d, device=%s\n' % \
          (type(model).__name__, type(optimizer).__name__,
           optimizer.param_groups[0]['lr'], epochs, device))
    
    #model.to(device)


    history             = {} # Collects per-epoch loss and acc like Keras' fit().
    history['loss']     = []
    history['val_loss'] = []
    history['acc']      = []
    history['val_acc']  = []

    start_time_sec = time.time()


    for epoch in range(1, epochs+1):

        # --- TRAIN AND EVALUATE ON TRAINING SET -----------------------------
        model.train()
        train_loss         = 0.0
        num_train_correct  = 0
        num_train_examples = 0

        for batch in train_dl:

            optimizer.zero_grad()

            x    = batch[0].to(device)
            y    = batch[1].to(device)
            yhat = model(x)
            loss = loss_fn(yhat, y)

            loss.backward()
            optimizer.step()

            train_loss         += loss.data.item() * x.size(0)
            num_train_correct  += (torch.max(yhat, 1)[1] == y).sum().item()
            num_train_examples += x.shape[0]

        train_acc   = num_train_correct / num_train_examples
        train_loss  = train_loss / len(train_dl.dataset)


        # --- EVALUATE ON VALIDATION SET -------------------------------------
        model.eval()
        val_loss         = 0.0
        num_val_correct  = 0
        num_val_examples = 0

        for batch in val_dl:

            x    = batch[0].to(device)
            y    = batch[1].to(device)
            yhat = model(x)
            loss = loss_fn(yhat, y)

            val_loss         += loss.data.item() * x.size(0)
            num_val_correct  += (torch.max(yhat, 1)[1] == y).sum().item()
            num_val_examples += y.shape[0]

        val_acc  = num_val_correct / num_val_examples
        val_loss = val_loss / len(val_dl.dataset)


        if epoch == 1 or epoch % 10 == 0:
            print('Epoch %3d/%3d, train loss: %5.2f, train acc: %5.2f, val loss: %5.2f, val acc: %5.2f' % \
            (epoch, epochs, train_loss, train_acc, val_loss, val_acc))

        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['acc'].append(train_acc)
        history['val_acc'].append(val_acc)

    # END OF TRAINING LOOP

    end_time_sec       = time.time()
    total_time_sec     = end_time_sec - start_time_sec
    time_per_epoch_sec = total_time_sec / epochs
    print()
    print('Time total:     %5.2f sec' % (total_time_sec))
    print('Time per epoch: %5.2f sec' % (time_per_epoch_sec))

    return history



#######################################################################
#                                                                     #
#             Proceeding of the extraction of the data                #
#                                                                     #
#######################################################################
 
mat_fname_test = pjoin(data_path, 'test_annotations.mat')
mat_contents_test = sio.loadmat(mat_fname_test)

mat_fname_train = pjoin(data_path, 'train_annotations.mat')
mat_contents_train = sio.loadmat(mat_fname_train)        
    
images_test , gazes_test   = dataset_extractor_test(data_path, mat_contents_test)

import pickle

with open('images.pkc', 'wb') as f1:
    for item in images_test:
        pickle.dump(item, f1)
 
with open('gazes.pkc', 'wb') as f2:
    for item in gazes_test:
        pickle.dump(item, f2)

#with open('/content/drive/MyDrive/MLA_Projet/images.pkc', 'rb') as f1:
#    images_test = pickle.load(f1)

#with open('/content/drive/MyDrive/MLA_Projet/gazes.pkc', 'rb') as f2:
#    gazes_test = pickle.load(f2)

#with open('/content/drive/MyDrive/depth_estimation_maps.rar (Unzipped Files)/depth_estimation_maps_gaze_follow.pkl', 'rb') as f3:
#    dual_att_map = pickle.load(f3)

#pre-processing
preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.double()),
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])
    
    
data_test = GazeFollowDataset(images_test, gazes_test, transform=preprocess)
        

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
#           Création du modèle "Heatmap Regression Head"              #
#                                                                     #
#######################################################################  

#device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')



class h_reg(nn.Module):
    def __init__(self):
        super(h_reg, self).__init__()
        
        #Resnet
        self.backbone = backbone
        
        #convolution layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=10, stride=2, padding=6)
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=6)

        #deconvolutional layer
        self.downsample1 = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1)
        self.downsample2 = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1)
        self.downsample3 = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1)
        
        
    # x represents our data
    def forward(self, x):

        x = self.backbone(x)
       
        x = x.unsqueeze(dim=1) 
        x = self.conv1(x)
        x = self.conv2(x)
        
        x = self.downsample1(x)
        x = self.downsample2(x)
        x = self.downsample3(x)
        
        x = x.squeeze(dim=1)

        output = F.softmax(x, dim=1)
        print(output.shape)
        output = output.reshape(1, 64, 64)
        
        
        return output

    

#Only a simple test on the Heatmap Regression Head with the CPU
#---------------------------------------------------------------

# Equates to one random 448x224 image
random_data = torch.rand((1, 3, 224+224, 224))
#random_data = random_data.to(device="cuda")

my_nn = h_reg()
#my_nn = my_nn.to(device="cuda")
result = my_nn(random_data)

#result = result.detach().numpy() 
print(result)

#plt.imshow(result[0,:,:])
#plt.colorbar()
#plt.show()


#Proceeding of the training

validation_split = .2
batch_size= 1
dataset_size = len(data_test)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)



network           = h_reg()
network           = network.double()
optimizer         = torch.optim.Adam(network.parameters(), lr = 0.001)
loss_fn           = nn.MSELoss()
epochs            = 2
#device            = "cuda"
train_loader      = DataLoader(data_test, batch_size=batch_size, sampler=train_sampler)
validation_loader = DataLoader(data_test, batch_size=batch_size, sampler=valid_sampler)

history = train(network, optimizer, loss_fn, train_loader, validation_loader, epochs, device)













    

