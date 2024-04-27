# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 09:52:48 2020

@author: 95883
"""

import torch
from torch import nn 
#import torch.nn.functional as F
#import cv2
import os
import glob
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
#import random
import numpy as np

from metrics import *

"""
from ptflops import get_model_complexity_info
from torchvision import models
"""

#加载数据
class Loader(Dataset):
    def __init__(self, data_path):
        #初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.img_path = glob.glob(os.path.join(data_path, 'train_image/*.raw')) #glob函数返回的是一个list
    """ 
    def augment(self, image, flipCode):
        #使用cv2.flip进行数据增强，当filpCode为1时水平翻转，0时垂直翻转，-1时水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip
    """
    def __getitem__(self, index):
        #根据index来读取图片
        image_path = self.img_path[index]
        #根据image_path来生成label_path
        label_path = image_path.replace('train', 'label')
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
        return len(self.img_path)

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

#U-Net结构
class DoubleConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
                torch.nn.BatchNorm2d(out_channels),  #BN层
                torch.nn.ReLU(),
                torch.nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(),
                #torch.nn.MaxPool2d(2)
                )
        
    def forward(self, x):
        return self.double_conv(x)

class MaxPool(torch.nn.Module):
    def __init__(self):
        super(MaxPool, self).__init__()
        self.maxpool = torch.nn.MaxPool2d(2)
        #self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size = 2, stride = 2)
        
    def forward(self, x):
        return self.maxpool(x)
  
class Down(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = torch.nn.Sequential(
                #torch.nn.MaxPool2d(2),
                DoubleConv(in_channels, out_channels)
                )
        
    def forward(self, x):
        return self.maxpool_conv(x)
    
class Up(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = torch.nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size = 2, stride = 2)
        self.conv = DoubleConv(in_channels, out_channels)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        #拼接，即U-Net结构中的skip-connection部分
        x = torch.cat([x2, x1], dim = 1)
        
        return self.conv(x)
 
class Last_upsample(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Last_upsample, self).__init__()
        self.last_up = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 2, stride = 2, padding = 2)
        
    def forward(self, x):
        x = self.last_up(x)
        
        return x

class OutConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size = 1)
        
    def forward(self, x):
        return self.conv(x)
    
#RNN模块
class RNN(torch.nn.Module):
    def __init__(self, input_size, output_size, in_channels, out_channels = 1):
        super(RNN, self).__init__()
        
        self.rnn = torch.nn.GRU(
                input_size = input_size,
                hidden_size = output_size,
                num_layers = 2,
                batch_first = True
                )
        
        self.cnn = torch.nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1)
        
        self.x = input_size
        self.y = output_size
        
    def forward(self, x):
        
        #in_channel = list(x.shape)[1]
        #x = x[:, -1, :, :]
        #x = x.view(-1, 512, 512)

        if list(x.shape)[1] != 1:
            x = self.cnn(x)
            
            x = torch.squeeze(x, 0)
            r_out, h_n = self.rnn(x)
        
        else:
            x = torch.squeeze(x, 0)
            r_out, h_n = self.rnn(x)
        #print("r_output: ")
        #print(r_out.shape)

        out = r_out[:, -1, :]  #out的三个维度分别为（batch_size, seq_legths, hidden_size）,[:, -1, :]这种形式将中间序列长度取-1，表示取序列中的最后一个数据，这个数据维度为512

        out = out.reshape(self.y // self.x, self.x)
        out = torch.unsqueeze(x, 0)
        #print("out: ")
        #print(out.shape)
        return out
    
class Channels(torch.nn.Module):
    def __init__(self, in_channels = 1, out_channels = 1):
        super(Channels, self).__init__()
        
        self.cnn = torch.nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)
        
    def forward(self, x):
        return self.cnn(x)
    


class UNet(torch.nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        self.maxpool = MaxPool()
        
        
        self.rnn1 = RNN(128, 15360, 1)#input_size, output_size, in_channels
        self.rnn2 = RNN(64, 3840, 64)
        self.rnn3 = RNN(32, 960, 128)
        self.rnn4 = RNN(16, 240, 256)
        #self.rnn5 = RNN(32, 32)
        
        self.cnn1 = Channels(1, 64)
        self.cnn2 = Channels(1, 128)
        self.cnn3 = Channels(1, 256)
        self.cnn4 = Channels(1, 512)
        
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(1, 128)
        self.down2 = Down(1, 256)
        self.down3 = Down(1, 512)
        self.down4 = Down(1, 1024)
        
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)
        #self.last_up = Last_upsample(2, 2)
        
    def forward(self, x):
        
        x_first = x / 255
        
        image1 = self.rnn1(x)
        #image1 = torch.unsqueeze(image1, 0)
        

        x1 = self.inc(image1)
        #print(type(x1))
        #print(x1.shape)
        x1_pool = self.maxpool(x1)
        
        image2 = self.rnn2(x1_pool)
        #image2 = torch.unsqueeze(image2, 0)
        #image2 = torch.unsqueeze(image2, 0)
        #image2 = self.cnn1(image2)
        
        #print(type(image2))
        #print(image2.shape)
        
        x2 = self.down1(image2)
        #print(x2.shape)
        
        x2_pool = self.maxpool(x2)
        
        image3 = self.rnn3(x2_pool)
        #image3 = torch.unsqueeze(image3, 0)
        #image3 = torch.unsqueeze(image3, 0)
        #image3 = self.cnn2(image3)
        
        
        x3 = self.down2(image3)
        #print(x3.shape)
        
        x3_pool = self.maxpool(x3)
        
        image4 = self.rnn4(x3_pool)
        #image4 = torch.unsqueeze(image4, 0)
        #image4 = torch.unsqueeze(image4, 0)
        #print(image4.shape)
        #image4 = self.cnn3(image4)
        #print(image4.shape)
        
        
        x4 = self.down3(image4)
        #print(x4.shape)
        
        #x4_pool = self.maxpool(x4)
        
        #image5 = self.rnn5(x4_pool)
        #image5 = torch.unsqueeze(image5, 0)
        #image5 = torch.unsqueeze(image5, 0)
        #image5 = self.cnn4(image5)
        
        #x5 = self.down4(image5)
        #print(x5.shape)
        #x = self.up1(x5, x4)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        x = self.outc(x)
        
        #x = self.last_up(x)
        #残差
        x = x + x_first
        
        return x

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
    
    print("complete")
    
    return loss_mean,metrics_acc

#加载网络，输入通道为1，类别为1（背景不算类别）
net = UNet(1, 1)
"""
ops, params = get_model_complexity_info(net, (1, 120, 128), as_strings = True,
                                        print_per_layer_stat = True, verbose = True)
"""
#采用GPU加速运算
device = torch.device("cuda") 
net = net.to(device) 


 
#选择Loss函数和优化器
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr = 0.0001, weight_decay = 1e-8)

#训练函数
def train(epochs, data_path):
    dataset = Loader(data_path)
    train_loader = DataLoader(dataset = dataset, batch_size = 1, shuffle = True)
    
    #best_loss统计，初始化为正无穷
    best_loss = float('inf')
    
    train_loss_acc = []
    multi_metric = []
    
    test_loss_acc = []
    test_metrics_acc = []
    
    for epoch in range(epochs):
        
        epochs_metric = []
        epochs_loss = []
        
        #test_loss,test_metrics = testset_loss(net)
        
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
            
        
        epochs_loss_mean = np.array(epochs_loss).mean()
        epochs_multi_metrics_mean = np.array(epochs_metric).mean(axis = 0)
        
        test_loss,test_metrics = testset_loss(net)
        
        print("Epoch: %d, Train Loss: %f, TEST LOSS:%f" % (epoch+1, epochs_loss_mean,test_loss))
        print("multi")
        print(epochs_multi_metrics_mean)
        #保存loss值最小的网络参数
        #if test_loss < best_loss:
        #    best_loss = test_loss
        #    torch.save(net.state_dict(), "best_model.path")
        
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
    
    return train_loss_acc,multi_metric,test_metrics_acc,test_loss_acc
            
#测试函数
def test():
    #加载训练好的模型参数
    #net.load_state_dict(torch.load('best_model.path', map_location = device))
    net.load_state_dict(torch.load('best_model.path'))
    #读取测试集中的图片路径
    tests_path = glob.glob('data/test/*.raw')
    
    net.eval() #测试模式，有BN层或Dropout层才有效
    
    #测试时不需要求梯度
    with torch.no_grad():
        for test_path in tests_path:
            #保存结果地址
            save_res_path = test_path.split('.')[0] + '_pred.raw'
            #读取图片
            """
            img = cv2.imread(test_path)
            """
            img = np.fromfile(test_path)
            img = img.reshape(120, 128)
            #转为灰度图
            """
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            """
            #转为batch为1，通道为1， 大小为512x512的数组
            img = img.reshape(1, 1, img.shape[0], img.shape[1])
            #转为tensor
            img_tensor = torch.from_numpy(img)
            #将tensor拷贝到device中
            img_tensor = img_tensor.to(device, dtype = torch.float32)
            #预测
            pred = net(img_tensor)
            #提取结果
            pred = np.array(pred.data.cpu()[0])[0]
            
            pred = pred * 255
            
            #cv2.imwrite(save_res_path, pred)
            pred.tofile(save_res_path)


            
if __name__ == "__main__":
    
    data_path = "data/"
    train(30, data_path) #训练20个epoch
    #test()
    print("运行完毕！！")
