#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 16:42:02 2022

@author: bring
"""

import torch
from torch import nn
import torch.nn.functional as F

class GRU(torch.nn.Module):
    def __init__(self, input_size, output_size, in_channels, out_channels = 1):
        super(GRU, self).__init__()
        
        self.rnn = torch.nn.GRU(
                input_size = input_size,#将横轴的一行数据送进去
                hidden_size = output_size//2,#双向
                num_layers = 2,
                batch_first = True,
                bidirectional = True
                )
        
        self.cnn = torch.nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1)
        
        self.x = input_size
        self.y = output_size
        
        return None
        
    def forward(self, x):
        
        #in_channel = list(x.shape)[1]
        #x = x[:, -1, :, :]
        #x = x.view(-1, 512, 512)


        if list(x.shape)[1] != 1:
            x = self.cnn(x)#这里可以试试用maxpool
            
            x = torch.squeeze(x, 0)
            r_out, h_n = self.rnn(x)
        
        else:
            x = torch.squeeze(x, 0)
            r_out, h_n = self.rnn(x)


        #print("r_output: ")
        #print(r_out.shape)

        out = r_out[:, -1, :]  #out的三个维度分别为（batch_size, seq_legths, hidden_size）,[:, -1, :]这种形式将中间序列长度取-1，表示取序列中的最后一个数据，这个数据维度为512

        out = out.reshape(self.y // self.x, self.x)
        out = F.tanh(torch.unsqueeze(x, 0))
        #print("out: ")
        #print(out.shape)
        return out
    
class Block1(torch.nn.Module):
    
    def __init__(self,input_size, output_size, in_channels,out_channels,gru_out_channels = 1):
        
        super(Block1,self).__init__()

        self.gru_module = GRU(input_size, output_size, in_channels,gru_out_channels)

        self.conv = nn.Conv2d(gru_out_channels,out_channels, stride = 1,padding=1,kernel_size=3)
        self.bn = nn.BatchNorm2d(out_channels)
        
        return None

    def forward(self,x):
        
        gru_value = self.gru_module(x)
        out = F.relu(self.bn(self.conv(gru_value)))
        
        return out

class Conv_block(torch.nn.Module):
    
    def __init__(self,in_channels,out_channels):
        
        super(Conv_block,self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels,out_channels, stride = 1,padding=1,kernel_size=3)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        return None
    
    def forward(self,x):
        
        return F.relu(self.bn1(self.conv1(x)))
        
class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1,
            stride=1, bias=False,)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x

class Downsample(torch.nn.Module):
    
    def __init__(self,in_channels,out_channels,):
        
        super(Downsample,self).__init__()
        
        self.maxpool = nn.MaxPool2d(3,stride=1,padding=1)#不改变尺寸
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=2,padding=1)#尺寸除以2
        
        self.conv2 = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=2,padding=1)
        self.conv3 = nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        return None
    
    def forward(self,x):
        
        x_max = x #注意梯度
        
        max_path = self.maxpool(x_max)
        max_out = self.bn1(self.conv1(max_path))
        
        conv_path = F.relu(self.bn2(self.conv2(x)))
        conv_out = self.bn3(self.conv3(conv_path))
        
        out = F.relu(max_out+conv_out)
        
        return out


class Level1(torch.nn.Module):
    
    def __init__(self,input_size, output_size, in_channels,b1_out_channels,gru_out_channels = 1):#in b1out = 1 32
        
    #输入为in_channels大小，输出为b1_out*2
    
        super(Level1,self).__init__()
        
        self.block_1 = Block1(input_size, output_size, in_channels,b1_out_channels,gru_out_channels = 1)
        self.conv_1 = Conv_block(b1_out_channels,b1_out_channels)
        self.root = Root(b1_out_channels*2,b1_out_channels*2,3,False)
        
        return None
    
    def forward(self,x):
        
        b_out = self.block_1(x)
        conv_out = self.conv_1(b_out)
        out = self.root(b_out,conv_out)
        
        return out
    
class Level2(torch.nn.Module):
    
    def __init__(self,input_size, output_size, in_channels,out_channels,first_bl1_channel,gru_out_channels = 1):#inchannels, outchannels = 32,448
        
    #输入in 输出outchannels
    
        super(Level2,self).__init__()
        
        self.level1 = Level1(input_size, output_size,in_channels,first_bl1_channel)
        self.block1 = Block1(input_size, output_size, first_bl1_channel*2,first_bl1_channel*2)
        self.conv1 = Conv_block(first_bl1_channel*2,first_bl1_channel*2)
        self.root = Root(out_channels,out_channels,3,False)
        
        return None
    
    def forward(self,x_down,last_ida=None):#level1_out_down是矩阵
        
        level1_out = self.level1(x_down)
        bl1_out = self.block1(level1_out)
        conv_out = self.conv1(bl1_out)
        
        if last_ida != None:
            out = self.root(last_ida,level1_out,bl1_out,conv_out)
        else:
            out = self.root(level1_out,bl1_out,conv_out)
        
        return out

class Level3(torch.nn.Module):
    
    def __init__(self,input_size, output_size, in_channels,l2_out_channels,root_channel,gru_out_channels = 1):
        
        super(Level3,self).__init__()
        
        self.level2 = Level2(input_size,output_size,in_channels,l2_out_channels,in_channels)
        self.level1 = Level1(input_size,output_size,l2_out_channels,l2_out_channels)#因为这里输出的是l2_out_channels*2
        
        self.block1 = Block1(input_size,output_size,l2_out_channels*2,l2_out_channels*2)
        self.conv1 = Conv_block(l2_out_channels*2,l2_out_channels*2)
        
        self.root = Root(root_channel,root_channel,3,False)
        
        return None
    
    def forward(self,x_down,last_ida):
        
        level2_out = self.level2(x_down)
        
        level1_out = self.level1(level2_out)
        
        bl1_out = self.block1(level1_out)
        conv_out = self.conv1(bl1_out)
        
        out = self.root(last_ida,level2_out,level1_out,bl1_out,conv_out)
        
        return out,level2_out

class Encoder(torch.nn.Module):
    
    def __init__(self):
        
        super(Encoder,self).__init__()
        
        self.level1 = Level1( 128, 15360, 1, 32)
        self.level2 = Level2(64, 3840, 32,448,64)
        self.level3 = Level3(32,960,224,1344,9856)
        
        self.down1 = Downsample(64,32)
        self.maxpool1 = torch.nn.MaxPool2d(3,stride = 1,padding = 1)
        self.conv1 = torch.nn.Conv2d(64,64,padding=1,kernel_size=3,stride=2)
        
        self.down2 = Downsample(448,224)
        self.maxpool2 = torch.nn.MaxPool2d(3,stride = 1,padding = 1)
        self.conv2 = torch.nn.Conv2d(448,448,padding=1,kernel_size=3,stride=2)
        
        return None
    
    def forward(self,x):
        
        l1_out = self.level1(x)#64x120x128
        l1_down = self.down1(l1_out)#32x60x64
        l1_ida = self.maxpool1(self.conv1(l1_out))        
        
        l2_out = self.level2(l1_down,l1_ida)
        l2_down = self.down2(l2_out)#224x30x32
        l2_ida = self.maxpool2(self.conv2(l2_out)) #448x30x32
        
        l3_out,l3_level2_out = self.level3(l2_down,l2_ida)#9856x30x32
        
        return l3_out,l2_out,l3_level2_out

class EncoderSmall(torch.nn.Module):
    
    def __init__(self):
        
        super(EncoderSmall,self).__init__()
        
        self.level1 = Level1( 128, 15360, 1, 8)
        self.level2 = Level2(64, 3840, 8,112,16)
        self.level3 = Level3(32,960,56,336,2464)
        
        self.down1 = Downsample(16,8)
        self.maxpool1 = torch.nn.MaxPool2d(3,stride = 1,padding = 1)
        self.conv1 = torch.nn.Conv2d(16,16,padding=1,kernel_size=3,stride=2)
        
        self.down2 = Downsample(112,56)
        self.maxpool2 = torch.nn.MaxPool2d(3,stride = 1,padding = 1)
        self.conv2 = torch.nn.Conv2d(112,112,padding=1,kernel_size=3,stride=2)
        
        return None
    
    def forward(self,x):
        
        l1_out = self.level1(x)#64x120x128
        l1_down = self.down1(l1_out)#32x60x64
        l1_ida = self.maxpool1(self.conv1(l1_out))        
        
        l2_out = self.level2(l1_down,l1_ida)
        l2_down = self.down2(l2_out)#224x30x32
        l2_ida = self.maxpool2(self.conv2(l2_out)) #448x30x32
        
        l3_out,l3_level2_out = self.level3(l2_down,l2_ida)#9856x30x32
        
        return l3_out,l2_out,l3_level2_out
    
class EncoderMini(torch.nn.Module):
    
    def __init__(self):
        
        super(EncoderMini,self).__init__()
        
        self.level1 = Level1( 128, 15360, 1, 8)
        self.level2 = Level2(64, 3840, 8,112,16)
        
        self.down1 = Downsample(16,8)
        self.maxpool1 = torch.nn.MaxPool2d(3,stride = 1,padding = 1)
        self.conv1 = torch.nn.Conv2d(16,16,padding=1,kernel_size=3,stride=2)
        
        self.down2 = Downsample(112,56)
        self.maxpool2 = torch.nn.MaxPool2d(3,stride = 1,padding = 1)
        self.conv2 = torch.nn.Conv2d(112,112,padding=1,kernel_size=3,stride=2)
        
        return None
    
    def forward(self,x):
        
        l1_out = self.level1(x)#64x120x128
        l1_down = self.down1(l1_out)#32x60x64
        l1_ida = self.maxpool1(self.conv1(l1_out))        
        
        l2_out = self.level2(l1_down,l1_ida)
        l2_down = self.down2(l2_out)#224x30x32
        l2_ida = self.maxpool2(self.conv2(l2_out)) #448x30x32
        
        return l2_out
    
class EncoderMini2(torch.nn.Module):
    
    def __init__(self):
        
        super(EncoderMini2,self).__init__()
        
        self.level1 = Level1( 128, 15360, 1, 8)
        self.level2_1 = Level2(64, 3840, 8,112,16)
        self.level2_2 = Level2(32,960,56,336,56)
        
        self.down1 = Downsample(16,8)
        self.maxpool1 = torch.nn.MaxPool2d(3,stride = 1,padding = 1)
        self.conv1 = torch.nn.Conv2d(16,16,padding=1,kernel_size=3,stride=2)
        
        self.down2 = Downsample(112,56)
        self.maxpool2 = torch.nn.MaxPool2d(3,stride = 1,padding = 1)
        self.conv2 = torch.nn.Conv2d(112,112,padding=1,kernel_size=3,stride=2)
        
        return None
    
    def forward(self,x):
        
        l1_out = self.level1(x)#64x120x128
        l1_down = self.down1(l1_out)#32x60x64
        l1_ida = self.maxpool1(self.conv1(l1_out))        
        
        l2_out = self.level2_1(l1_down,l1_ida)
        l2_down = self.down2(l2_out)#224x30x32
        l2_ida = self.maxpool2(self.conv2(l2_out)) #448x30x32
        
        l3_level2_out = self.level2_2(l2_down)
        
        return l2_out,l3_level2_out
        

if __name__ == "__main__":
    
    encoder = EncoderMini2()
    vec = torch.rand([1,1,120,128])
    reuslt = encoder(vec)





        
        
        
        
        
        







































