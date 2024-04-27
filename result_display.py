#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 10:23:42 2022

@author: bring
"""
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import glob
import numpy as np

from Loader import *
#from Net import *
from DLA_UNet import *
#from IHAGUNET_CAT import UNetVer2,UNetPrevious
#from IHAGUNET_ADD import UNet

import torch
from torchvision import transforms

def net_loader(model_path,name):
    
    if name == 'DLANet':
        net = NetworkSmall()
    elif name == 'IHAGUNet_ADD':
        net= UNet(1,1)
    elif name == 'IHAGUNet_CAT':
        net = UNetVer2(1,1)
    elif name == 'LUNet':
        net = UNetPrevious(1,1)
    elif name == 'REDCNNNet':
        net = REDCNNNet()
    elif name == 'DLA_UNet_Third':
        net = DLA_UNet_Third(1,1)
    elif name == 'DLA_UNet_Second':
        net = DLA_UNet_Second(1,1)
    else:
        print('wrong')

    net.load_state_dict(torch.load(model_path+'best_model_{}_final.path'.format(name), map_location=torch.device('cpu')))
    #net.eval()
    return net

def tensor_to_PIL(tensor,path):
        
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    image.save(path)
        
    return image

def convert_to_jpg(data_path):
        #,'IHAGUNet_ADD',
     net_list = ['DLA_UNet_Second']#['IHAGUNet_CAT', ,'REDCNNNet'],'LUNet''DLANet2'
    
     train_path = glob.glob(os.path.join(data_path, 'test_image/*.raw')) 
     label_path = glob.glob(os.path.join(data_path, 'test_target/*.raw'))


     for j in net_list:
         
         net = net_loader(r'./net/',j)

         for i in range(len(train_path[:100])):
             
             image = np.fromfile(train_path[i], dtype = np.float64)
             #label = np.fromfile(label_path[i], dtype = np.float64)
             
             image = image.reshape(120, 128)
             #label = label.reshape(120, 128)
             
             image = torch.FloatTensor(image.reshape(1, image.shape[0], image.shape[1]))
             #label = torch.FloatTensor(label.reshape(1, label.shape[0], label.shape[1]))
         
             image = net(image.unsqueeze(0))
             
             image = (image - image.min())/(image.max()-image.min())
             
             #image[image<0]=0
             #image[image>255] = 255
         
             tensor_to_PIL(image,r'./data/{}_jpg/test{}.png'.format(j,train_path[i].split('/')[-1]))
             #tensor_to_PIL(image,r'./data/train_jpg/{}.jpg'.format(label_path[i].split('/')[-1]))
             #tensor_to_PIL(label,r'./data/test_jpg/{}.jpg'.format(label_path[i].split('/')[-1]))
             
     return None

def convert_to_jpg1(data_path):
    
     train_path = glob.glob(os.path.join(data_path, 'test_image/*.raw')) 
     label_path = glob.glob(os.path.join(data_path, 'test_target/*.raw'))

     for i in range(len(train_path)):
             
         image = np.fromfile(train_path[i], dtype = np.float64)
         label = np.fromfile(label_path[i], dtype = np.float64)
             
         image = image.reshape(120, 128)
         label = label.reshape(120, 128)
             
         image = torch.FloatTensor(image.reshape(1, image.shape[0], image.shape[1]))
         label = torch.FloatTensor(label.reshape(1, label.shape[0], label.shape[1]))
         
         image = (image - image.min())/(image.max()-image.min())
         label = (label-label.min())/(label.max()-label.min())

         #tensor_to_PIL(image,r'./data/_jpg/{}.png'.format(label_path[i].split('/')[-1]))
         tensor_to_PIL(image,r'./data/test_image_jpg/{}.jpg'.format(label_path[i].split('/')[-1]))
         tensor_to_PIL(label,r'./data/test_label_jpg/{}.jpg'.format(label_path[i].split('/')[-1]))
             
     return None

if __name__ == '__main__':
    
    
    convert_to_jpg(r'./data/')