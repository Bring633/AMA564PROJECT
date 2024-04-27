#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 16:38:53 2022

@author: bring
"""

import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import glob
import numpy as np

class Loader(Dataset):
    def __init__(self, data_path):
        #初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.train_path = glob.glob(os.path.join(data_path, 'train_image/*.raw')) #glob函数返回的是一个list
        self.label_path = glob.glob(os.path.join(data_path, 'label_image/*.raw'))
    """ 
    def augment(self, image, flipCode):
        #使用cv2.flip进行数据增强，当filpCode为1时水平翻转，0时垂直翻转，-1时水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip
    """
    def __getitem__(self, index):
        #根据index来读取图片
        image_path = self.train_path[index]
        #根据image_path来生成label_path
        label_path = self.label_path[index]
        #读取训练图片和标签图片
        """
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)
        """
        image = np.fromfile(image_path, dtype = np.float64)
        label = np.fromfile(label_path, dtype = np.float64)
        image = image.reshape(120, 128)#为什么转成这样？
        label = label.reshape(120, 128)
        #将数据转为单通道图片
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        """
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        #处理标签，将像素值为255的改为1
        
        if label.max() > 1:
            label = label / 255
            image = image / 255
        
        #随机进行数据增强，为2时不做处理
        #flipCode = random.choice([-1, 0, 1, 2])
        """
        flipCode = 2
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)
        """
        return image, label
    
    def __len__(self):
        return len(self.train_path)
    
class TestLoader(Dataset):
    def __init__(self, data_path):
        #初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.train_path = glob.glob(os.path.join(data_path, 'test_image/*.raw')) #glob函数返回的是一个list
        self.label_path = glob.glob(os.path.join(data_path, 'test_target/*.raw'))

    def __getitem__(self, index):
        #根据index来读取图片
        image_path = self.train_path[index]
        #根据image_path来生成label_path
        label_path = self.label_path[index]
        #读取训练图片和标签图片

        image = np.fromfile(image_path, dtype = np.float64)
        label = np.fromfile(label_path, dtype = np.float64)
        image = image.reshape(120, 128)#为什么转成这样？
        label = label.reshape(120, 128)
        #将数据转为单通道图片

        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        
        #处理标签，将像素值为255的改为1
        if label.max() > 1:
            label = label / 255
            image = image / 255
        
        #随机进行数据增强，为2时不做处理
        #flipCode = random.choice([-1, 0, 1, 2])
 
        return image, label
    
    def __len__(self):
        return len(self.train_path)