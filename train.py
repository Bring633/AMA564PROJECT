#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 16:35:58 2022

@author: bring
"""

import torch
from torch import nn

import os
import glob
from torch.utils.data import DataLoader
from torchvision import transforms
import random

import numpy as np
import pandas as pd

from Net import *
from UNet import *               
from DLA_UNet import *
from ALL_UNet import *
from DLA_UNet_asy import *
from Loader import Loader,TestLoader
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
from metrics import multi_metrics

import time

#from thop import profile
#from pytorch_model_summary import summary
#import matplotlib.pyplot as plt

def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.

seed_torch()


data_path = r'./data/'
model_path = r'./'

device =  torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#device = torch.device("cuda:1")
#device = torch.device"cpu")

#网络设置
#net = R2U_Net(1,1).to(device) 
#net = REDCNNNet().to(device)
net = DLA_UNet_asy(1,1).to(device)
#net = UNet(1,1).to(device)
#net  = NestedUNet(1,1).to(device)#
#net = DLA_UNet(1,1).to(device)
#net.load_state_dict(torch.load(model_path+'best_model_{}_final.path'.format(net.__class__.__name__)))

#优化器设置
learning_rate = 0.0002
wd = 0.01#3\e-12

optimizer = torch.optim.AdamW(net.parameters(), lr = learning_rate,weight_decay=wd)

#选择Loss函数
criterion = torch.nn.MSELoss()

def tensor_to_PIL(tensor,path):
        
    
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()
    image = (image - image.min())/(image.max()-image.min())
    image = image.squeeze(0)
    image = unloader(image)
    image.save(path)
        
    return image

#损失曲线
def plot_curve(train_data,test_data):
    
    plt.figure()
    plt.plot(range(len(train_data)),train_data)
    plt.plot(range(len(test_data)),train_data)
    plt.title("Train result visualization")
    plt.show()
    
    return None


#测试集上的误差
def testset_loss(net,test_path=r'./data/'):
    
    net.eval()
    
    dataset = TestLoader(test_path)
    test_loader = DataLoader(dataset = dataset, batch_size = 1, shuffle = True)
    
    criterion = nn.MSELoss()
    loss_acc = []
    metrics_acc = []
    
    for image, label in test_loader:
        
        image, label = image.to(device, dtype = torch.float32), label.to(device, dtype = torch.float32)
        outs = net(image)
        loss = criterion(outs, label)
    
        multi_metric_result = multi_metrics(outs,label)
        metrics_acc.append(multi_metric_result)
        loss_acc.append(loss.cpu().item())
        
    loss_mean = np.array(loss_acc).mean()
    metrics_acc = np.array(metrics_acc).mean(axis = 0)
    
    net.train()
    
    return loss_mean,metrics_acc

#写文件
def write_data(train_loss_acc,multi_metric,test_metrics_acc,test_loss_acc,data_path=r'./data/'):

    df = pd.DataFrame([train_loss_acc,multi_metric,test_metrics_acc,test_loss_acc])
    df.to_csv(r'./result_{}.csv'.format(time.time()))
    
    return None



#训练函数
def train(epochs, data_path):
    dataset = Loader(data_path)
    train_loader = DataLoader(dataset = dataset, batch_size = 1, shuffle = True)
    
    #best_loss统计，初始化为正无穷
    best_loss = float('0.0')
    
    train_loss_acc = []
    multi_metric = []
    
    test_loss_acc = []
    test_metrics_acc = []

    for epoch in range(epochs):
        
        epochs_metric = []
        epochs_loss = []
        
        for image, label in train_loader:
            
            #GPU加速运算
            image, label = image.to(device, dtype = torch.float32), label.to(device, dtype = torch.float32)
            """
            #LSTM增加的部分操作，因为lstm的size是（batch, seqlen, input_size）,而u-net的size是(batch, channel, W, H)因此需要做维度的转换
            image_lstm = image.view(-1, 512, 512)
            r_output, image = rnn(image_lstm)
            image = torch.unsqueeze(image, 0)
            image = torch.unsqueeze(image, 0)
            """
            optimizer.zero_grad()
            outs = net(image)
            
            loss = criterion(outs, label)
            
            loss.backward()
            optimizer.step()
            
            multi_metric_result = multi_metrics(outs,label)
            epochs_metric.append(multi_metric_result)
            epochs_loss.append(loss.cpu().item())
            
            #print("Train Loss: %f" % (loss))
            #print(multi_metric_result)
            
        
        tensor_to_PIL(outs,r'./training.png')
        
        epochs_loss_mean = np.array(epochs_loss).mean()
        epochs_multi_metrics_mean = np.array(epochs_metric).mean(axis = 0)
        
        test_loss,test_metrics = testset_loss(net)
        
        print("Epoch: %d, Train Loss: %f, Test Loss: %f" % (epoch+1, epochs_loss_mean,test_loss))
        print("multi")
        print(epochs_multi_metrics_mean)
        print(test_metrics)
        #保存loss值最小的网络参数
        if test_metrics[1] > best_loss:
            best_loss = test_metrics[1]
            torch.save(net.state_dict(), "best_model_{}.path".format(net.__class__.__name__))
        
        train_loss_acc.append(epochs_loss_mean)
        multi_metric.append(epochs_multi_metrics_mean)
        
        test_loss_acc.append(test_loss)
        test_metrics_acc.append(test_metrics)
        
        
    #plot_curve(train_loss_acc,test_loss_acc)
    try:
        write_data(train_loss_acc,multi_metric,test_metrics_acc,test_loss_acc)
    except:
        
        print(train_loss_acc)
        print(multi_metric)
        print(test_metrics_acc)
        print(test_loss_acc)
        
        return None
    
    torch.save(net.state_dict(), "best_model_{}_final.path".format(net.__class__.__name__))
    
    return train_loss_acc,multi_metric,test_metrics_acc,test_loss_acc
    
            
if __name__ == "__main__":
    
    train(80,data_path)
    
    #dummy_input = torch.randn(1, 1, 120, 128)
    #flops, params = profile(net, (dummy_input,))
    #dummy_input = torch.randn(1, 1, 120, 128)
    #print(summary(net, dummy_input, show_input=False, show_hierarchical=False))
    #print('FLOPs: ', flops, 'params: ', params)

