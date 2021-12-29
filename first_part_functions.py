from torchvision import models 
from torchvision import transforms
import torch.nn as nn
from PIL import Image
from matplotlib import cm
import torch
from torch.utils.data.sampler import SubsetRandomSampler
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import numpy as np
from os.path import dirname, join as pjoin
import scipy.io as sio
import cv2
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.patches as patches
import math
from math import *
from torch.utils.data import Dataset
import torchvision
import torch.optim as optim

#first check  of eye detection
def  Kellnhofer_al_eye(imHead,headBBInFull,eyeBBInFull): 
        cropSizePx = [imHead.shape[1], imHead.shape[0]]
        #eyeBBInFull = person_eye_left_bbox[i,:]
        eyeBBInCrop = [
            (eyeBBInFull[0] - headBBInFull[0]) / headBBInFull[2], # subtract offset of the crop
            (eyeBBInFull[1] - headBBInFull[1]) / headBBInFull[3], 
            eyeBBInFull[2] / headBBInFull[2], # scale to smaller space of the crop
            eyeBBInFull[3] / headBBInFull[3], 
            ]
        eyeBBInCropPx_l = np.concatenate([np.multiply(eyeBBInCrop[:2], cropSizePx), np.multiply(eyeBBInCrop[2:], cropSizePx)]).astype(int)
        imEye_l = imHead[
            eyeBBInCropPx_l[1]:(eyeBBInCropPx_l[1]+eyeBBInCropPx_l[3]), 
            eyeBBInCropPx_l[0]:(eyeBBInCropPx_l[0]+eyeBBInCropPx_l[2]),
            :]
        fl= imEye_l.flatten()
        imEye_l = cv2.resize(imEye_l,(60,36)) 
        return imEye_l


#1st check of eye detection : using the technique of kellnhofer et al article, detect eyes if possible, else, return zeros 
def Kellnhofer_al_eyes(imHead,head_box,eye_left_box,eye_right_box):
    if np.all(eye_left_box==np.array([-1,-1,-1,-1])):
        imEye_l=np.zeros((36,60,3), np.uint8)
    else:
        imEye_l = Kellnhofer_al_eye(imHead,head_box,eye_left_box)

    if np.all(eye_right_box==np.array([-1,-1,-1,-1])):
        imEye_r= np.zeros((36,60,3), np.uint8)
    else:
        imEye_r = Kellnhofer_al_eye(imHead,head_box,eye_right_box)
    #return fl,fr
    return imEye_l,imEye_r

#second check : using the 3D technique of bulat et al article, detect eyes, else return zeros
def bulat_al(imHead,fa):#second check
    try : 
      preds = fa.get_landmarks(imHead)[-1]#predict
      # 2D-Plot
      pred_type = collections.namedtuple('prediction_type', ['slice', 'color'])
      pred_types = {
                'eye1': pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
                'eye2': pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
                }
      centers=[]
      Avr_eye=24 #24 represents average eye length for adults, we use this to set the scale

      for pred_type in pred_types.values():
          x=preds[pred_type.slice, 0]
          y=preds[pred_type.slice, 1]
          centroid = (sum(x) / len(x), sum(y) / len(y)) #get centroid
          centers.append(centroid) #append
      dist = math.hypot(centers[0][0]-centers[1][0],centers[0][1]-centers[1][1]) #distance on image
      dist_reel= np.divide(dist*24,x.max()-x.min() )#real distance
      if (77 >dist_reel>51):  #normal distance between pupils is between 51 and 77
          centers = [(int(element[0]), int(element[1])) for element in centers]
          imEye_r=imHead[centers[0][0]-7:centers[0][0]+7,centers[0][1]-7:centers[0][1]+7,:]
          imEye_l=imHead[centers[1][0]-7:centers[1][0]+7,centers[1][1]-7:centers[1][1]+7,:]
          fl= imEye_l.flatten()
          fr= imEye_r.flatten()
          imEye_r = cv2.resize(imEye_r,(60,36)) 
          imEye_l = cv2.resize(imEye_l,(60,36)) 
      else :
          blank_image = np.zeros((36,60,3), np.uint8)
          imEye_l = blank_image
          imEye_r = blank_image
    except : 
      blank_image = np.zeros((36,60,3), np.uint8)
      imEye_l = blank_image
      imEye_r = blank_image
    return imEye_l,imEye_r

# extract eyes from head image function
def head_eye_extractors(imHead,fa,head_box,target,eye_left_box,eye_right_box):
      if np.all(eye_left_box==np.array([-1,-1,-1,-1])) and np.all(eye_right_box==np.array([-1,-1,-1,-1])):
          #print("eye_not_detected with Kellnhofer_al ")
          imEye_l,imEye_r= bulat_al(imHead,fa)
      else: 
          imEye_l,imEye_r= Kellnhofer_al_eyes(imHead,head_box,eye_left_box,eye_right_box)
      x,y,z=target
      yaw = atan(np.divide(z,x))
      pitch=atan(np.divide(x,y))
      h=[float(yaw),float(pitch)]
      return h, imEye_l,imEye_r

