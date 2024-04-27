#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 20:02:21 2022

@author: bring
"""

import torch
import numpy as np
import numpy
#from scipy import signal
#from scipy import ndimage
#from skimage.metrics import structural_similarity as ssim

#from sklearn.metrics import mean_squared_error

#from torch.nn import MSELoss

def NMSE(result,label):

    #(batch,channel,x,y)
    result = result.detach().cpu().numpy()
    label = label.detach().cpu().numpy()
    
    result = (result - np.min(result)) / (np.max(result) - np.min(result)) * 255
    #label = (label - np.min(label)) / (np.max(label) - np.min(label)) * 255
    label = label*255
    
    result =result / np.average(result) * np.average(label)
    
    mse = np.mean((result - label) ** 2)
    g_i = label**2
    g_i_sum = g_i.sum()
    
    return mse/g_i_sum

def PSNR(result,label):
    
    result = result.detach().cpu().numpy()
    label = label.detach().cpu().numpy()
    
    result = (result - np.min(result)) / (np.max(result) - np.min(result)) * 255
    #label = (label - np.min(label)) / (np.max(label) - np.min(label)) * 255
    label = label*255
    
    result =result / np.average(result) * np.average(label)
    
    mse = np.mean((result - label) ** 2)
    
    #result = 10*np.log10(max_**2/loss)
    result = 10*np.log10(255**2/mse)
    return result


def fspecial_gauss(size, sigma):
    
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()


def SSIM(result, label, cs_map=False):

    K1=0.01    
    K2=0.03
    L=255
    C1 = (K1*L)**2
    C2=(K2*L)**2

    if isinstance(result,torch.cuda.FloatTensor):
        result = np.array(result.detach().cpu())
        label = np.array(label.detach().cpu())  
    
    result = (result - np.min(result)) / (np.max(result) - np.min(result)) * 255
    #label = (label - np.min(label)) / (np.max(label) - np.min(label)) * 255
    label = label*255
    
    result =result / np.average(result) * np.average(label)
    
    result_mean = result.mean()
    label_mean = label.mean()

    try:
        result_std = np.std(result)
        label_std = np.std(label)
    
    except:
        print(result)
        print(result.shape)
        
        a = torch.cuda.FloatTensor([0.0])
        return a
    
    cov_result_label = np.cov(np.array([result.reshape(-1),label.reshape(-1)]))[0][1]
    
    ssim = (2*result_mean*label_mean+C1)*(2*cov_result_label+C2)/\
        ((result_mean**2+label_mean**2+C1)*(result_std**2+label_std**2+C2))

    #ssim_ = ssim(result, label, data_range=255)
    
    return ssim


def multi_metrics(result,label):
    
    nmse = NMSE(result,label).item()
    psnr = PSNR(result,label).item()
    ssim = SSIM(result,label).item()
    
    return [nmse,psnr,ssim]

if __name__ == '__main__':
    
    result = torch.rand([1,1,120,128])
    label = torch.rand([1,1,120,128])
    multi_metrics(result,label)
