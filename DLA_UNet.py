#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 20:55:01 2022

@author: bring
"""

import torch
# import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import numpy as np
from torch.autograd import Variable
from torch.nn.utils import weight_norm


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        output = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        pred = self.linear(output[:, -1, :])
        return pred


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
    
class GRU(torch.nn.Module):
    def __init__(self, input_size, output_size, in_channels, encoder,out_channels = 1):
        super(GRU, self).__init__()
        
        self.rnn = torch.nn.GRU(
                input_size = input_size,#将横轴的一行数据送进去
                hidden_size = output_size//1,#双向2 单向1
                num_layers = 2,
                batch_first = True,
                bidirectional = False
                )
        
        self.cnn1 = torch.nn.Conv2d(in_channels = in_channels, out_channels = in_channels, kernel_size = 1)
        self.cnn2 = torch.nn.Conv2d(in_channels = 120, out_channels = out_channels, kernel_size = 1)
        
        self.x = input_size
        self.y = output_size
        
        self.encoder = encoder
        
        return None
        
    def forward(self, x):
        
        #in_channel = list(x.shape)[1]
        #x = x[:, -1, :, :]
        #x = x.view(-1, 512, 512)


        if list(x.shape)[1] != 1:
            x = self.cnn1(x)#这里可以试试用maxpool
            
            x = torch.squeeze(x, 0)
            r_out, h_n = self.rnn(x)
        
        else:
            x = torch.squeeze(x, 0)
            r_out, h_n = self.rnn(x)


        #print("r_output: ")
        #print(r_out.shape)
        #out的三个维度分别为（batch_size, seq_legths, hidden_size）,[:, -1, :]这种形式将中间序列长度取-1，表示取序列中的最后一个数据，这个数据维度为512

        if self.encoder:
            out = r_out.squeeze().reshape(-1,r_out.shape[1],self.x//2)
            out = F.tanh(torch.unsqueeze(out, 0))
            out = self.cnn2(out)
        else:
            out = r_out
        #print("out: ")
        #print(out.shape)
        return out

class ConvDoubleBlock(nn.Module):
    def __init__(self, in_num_ch, out_num_ch,dropout, filter_size=3):
        super(ConvDoubleBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_num_ch, out_num_ch, filter_size, padding=1),
            nn.BatchNorm2d(out_num_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(out_num_ch, out_num_ch, filter_size, padding=1),
            nn.BatchNorm2d(out_num_ch),
            nn.ReLU(inplace=True),
            #nn.Dropout(dropout),
            )

    def forward(self, x):
        x = self.conv(x)
        return x

class Conv_block(torch.nn.Module):
    
    def __init__(self,in_channels,out_channels):
        
        super(Conv_block,self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels,out_channels, stride = 1,padding=1,kernel_size=3)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        return None
    
    def forward(self,x):
        
        return F.relu(self.bn1(self.conv1(x)))

#1
class DownBlock(nn.Module):
    def __init__(self, in_num_ch, out_num_ch,dropout=0):
        super(DownBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvDoubleBlock(in_num_ch, out_num_ch, dropout,3)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Down_2(torch.nn.Module):
    
    def __init__(self,in_channels,out_channels):
        
        super(Down_2,self).__init__()
        
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

class Down(torch.nn.Module):
    
    def __init__(self,in_channels,out_channels,time):
        
        super(Down,self).__init__()
        
        self.net = nn.Sequential()
        
        """ Second
        if time == 1:
            self.net.add_module('down_{}'.format(1),Down_2(in_channels,out_channels))
            return None
    
        for i in range(0,time):
            if i == time-1:
                self.net.add_module('down_{}'.format(i),Down_2(in_channels*2**i,out_channels))
            else:
                self.net.add_module('down_{}'.format(i),Down_2(in_channels*2**i,in_channels*2**(i+1)))
        """
        if time == 1:
            self.net.add_module('down_{}'.format(1),Down_2(in_channels,out_channels))
            return None
    
        for i in range(0,time):
            if i == time-1:
                self.net.add_module('down_{}'.format(i),Down_2(in_channels*2**i,out_channels))
            else:
                self.net.add_module('down_{}'.format(i),Down_2(in_channels*2**i,in_channels*2**(i+1)))
        
        return None
    
    def forward(self,x):
        
        return self.net(x)

class UpBlock(nn.Module):
    def __init__(self, in_num_ch, out_num_ch, dropout,upsample=True):
        super(UpBlock, self).__init__()
        if upsample == True:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_num_ch, in_num_ch//2, 2)
                )
        else:
            self.up = nn.ConvTranspose2d(in_num_ch, in_num_ch//2, 3, padding=1, stride=2) # (H-1)*stride-2*padding+kernel_size
        #1
        self.conv = ConvDoubleBlock(in_num_ch, out_num_ch, dropout,3)

    def forward(self, x_down, x_up):
        
        x_up = self.up(x_up)
        diffY = (x_down.size()[2] - x_up.size()[2])
        diffX = (x_down.size()[3] - x_up.size()[3])
        x_up = F.pad(x_up, (diffX // 2, diffX - diffX//2, diffY // 2, diffY - diffY//2))  
        x = torch.cat([x_up, x_down], dim=1)
        x = self.conv(x)
    
        """
        x_up = self.up(x_up)
        # after the upsample/convtrans, the HW is smaller than the down-sampling map
        x_up = F.pad(x_up, (3//2,int(3/2),3//2,int(3/2)), mode='replicate')
        x = torch.cat([x_down, x_up], 1)
        x = self.conv(x)
        """
        return x



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

class DLA_UNet_Second(nn.Module):
    def __init__(self, in_num_ch, out_num_ch, first_num_ch=64, input_size=256, output_activation='softplus'):
        super(DLA_UNet_Second, self).__init__()
        self.down_1 = ConvDoubleBlock(in_num_ch, first_num_ch, 3)
        self.down_2 = DownBlock(first_num_ch, 2*first_num_ch)
        self.down_3 = DownBlock(2*first_num_ch, 4*first_num_ch)
        self.down_4 = DownBlock(4*first_num_ch, 8*first_num_ch)
        self.down_5 = DownBlock(8*first_num_ch, 16*first_num_ch)
        
        self.down_ida_1 = Conv_block(in_num_ch,first_num_ch)
        self.down_ida_2 = Down(first_num_ch,first_num_ch*2,1)
        self.down_ida_3 = Down(in_num_ch,first_num_ch*2,1)
        
        self.root_1 = Root(first_num_ch*2,first_num_ch,3,0)
        self.root_2 = Root(6*first_num_ch,2*first_num_ch,3,0)
        
        self.up_4 = UpBlock(16*first_num_ch, 8*first_num_ch)
        self.up_3 = UpBlock(8*first_num_ch, 4*first_num_ch)
        self.up_2 = UpBlock(4*first_num_ch, 2*first_num_ch)
        self.up_1 = UpBlock(2*first_num_ch, first_num_ch)
        self.output = nn.Conv2d(first_num_ch, out_num_ch, 1)
        # choose different activation layer
        if output_activation == 'sigmoid':
            self.output_act = nn.Sigmoid()
        elif output_activation == 'tanh':
            self.output_act = nn.Tanh()
        elif output_activation == 'linear':
            self.output_act = nn.Linear()
        else:
            self.output_act = nn.Softplus()

    def forward(self, x):
        
        down_1 = self.down_1(x)
        down_ida_1 = self.down_ida_1(x)
        
        down_2 = self.down_2(self.root_1(down_1,down_ida_1))
        down_ida_2 = self.down_ida_2(down_1)
        
        down_ida_3 = self.down_ida_3(x)
        down_3 = self.down_3(self.root_2(down_2,down_ida_2,down_ida_3))
        
        down_4 = self.down_4(down_3)
        down_5 = self.down_5(down_4)
        up_4 = self.up_4(down_4, down_5)
        up_3 = self.up_3(down_3, up_4)
        up_2 = self.up_2(down_2, up_3)
        up_1 = self.up_1(down_1, up_2)
        output = self.output(up_1)
        output_act = self.output_act(output)
        return output_act
    
class DLA_UNet_Third(nn.Module):
    def __init__(self, in_num_ch, out_num_ch, first_num_ch=64, input_size=256, output_activation='softplus'):
        super(DLA_UNet_Third, self).__init__()
        self.down_1 = ConvDoubleBlock(in_num_ch, first_num_ch, 3)
        self.down_2 = DownBlock(first_num_ch, 2*first_num_ch)
        self.down_3 = DownBlock(2*first_num_ch, 4*first_num_ch)
        self.down_4 = DownBlock(4*first_num_ch, 8*first_num_ch)
        self.down_5 = DownBlock(8*first_num_ch, 16*first_num_ch)
        
        self.down_ida_1 = Conv_block(in_num_ch,first_num_ch)
        self.down_ida_2 = Down(first_num_ch,first_num_ch*2,1)
        self.down_ida_3 = Down(in_num_ch,first_num_ch*2,1)
        self.down_ida_4 = Down(first_num_ch*2,first_num_ch*4,1)
        self.down_ida_5 = Down(first_num_ch*2,first_num_ch*4,2)
        self.down_ida_6 = Down(first_num_ch*4,first_num_ch*8,1)
        
        self.root_1 = Root(first_num_ch*2,first_num_ch,3,0)
        self.root_2 = Root(6*first_num_ch,2*first_num_ch,3,0)
        self.root_3 = Root(first_num_ch*8,first_num_ch*4,3,0)
        self.root_4 = Root(first_num_ch*20,first_num_ch*8,3,0)
        
        self.up_4 = UpBlock(16*first_num_ch, 8*first_num_ch)
        self.up_3 = UpBlock(8*first_num_ch, 4*first_num_ch)
        self.up_2 = UpBlock(4*first_num_ch, 2*first_num_ch)
        self.up_1 = UpBlock(2*first_num_ch, first_num_ch)
        self.output = nn.Conv2d(first_num_ch, out_num_ch, 1)
        # choose different activation layer
        if output_activation == 'sigmoid':
            self.output_act = nn.Sigmoid()
        elif output_activation == 'tanh':
            self.output_act = nn.Tanh()
        elif output_activation == 'linear':
            self.output_act = nn.Linear()
        else:
            self.output_act = nn.Softplus()

    def forward(self, x):
        
        down_1 = self.down_1(x)
        down_ida_1 = self.down_ida_1(x)
        
        down_2 = self.down_2(self.root_1(down_1,down_ida_1))
       
        down_ida_2 = self.down_ida_2(down_1)
        down_ida_3 = self.down_ida_3(x)
        
        down_3 = self.down_3(self.root_2(down_2,down_ida_2,down_ida_3))
        down_ida_4 = self.down_ida_4(down_2)
        down_4 = self.down_4(self.root_3(down_3,down_ida_4))
        
        down_ida_5 = self.down_ida_5(down_2)
        down_ida_35 = self.down_ida_6(down_3)
        down_5 = self.down_5(self.root_4(down_4,down_ida_5,down_ida_35))
    
        up_4 = self.up_4(down_4, down_5)
        up_3 = self.up_3(down_3, up_4)
        up_2 = self.up_2(down_2, up_3)
        up_1 = self.up_1(down_1, up_2)
        output = self.output(up_1)
        output_act = self.output_act(output)
        return output_act
    
    
class DLA_UNet(nn.Module):
    def __init__(self, in_num_ch, out_num_ch, first_num_ch=64, input_size=256, output_activation='softplus'):
        super(DLA_UNet, self).__init__()
        
        self.tcn = TCN(128,15360,[1,2,3],2,0.2)
        self.dc1 = ConvDoubleBlock(1,4,0.1)
        self.dc2 = ConvDoubleBlock(4,8,0.05)
        self.dc3 = ConvDoubleBlock(8,1,0.1)
        #self.rnn2 = GRU(128, 128,1,False,1)
        
        #1
        self.down_1 = ConvDoubleBlock(in_num_ch, first_num_ch,0.2)
        self.down_2 = DownBlock(first_num_ch, 2*first_num_ch,0.15)
        self.down_3 = DownBlock(2*first_num_ch, 4*first_num_ch,0.1)
        self.down_4 = DownBlock(4*first_num_ch, 8*first_num_ch,0.05)
        self.down_5 = DownBlock(8*first_num_ch, 16*first_num_ch,0.00)
        
        self.down_ida_1 = Conv_block(in_num_ch,first_num_ch)
        self.down_ida_2 = Down(first_num_ch,first_num_ch*2,1)
        self.down_ida_3 = Down(in_num_ch,first_num_ch*2,1)
        self.down_ida_4 = Down(first_num_ch*2,first_num_ch*4,1)
        self.down_ida_5 = Down(first_num_ch*2,first_num_ch*4,2)
        self.down_ida_6 = Down(first_num_ch*4,first_num_ch*8,1)
        self.down_ida_7 = Down(in_num_ch,first_num_ch*4,3)
        
        self.root_1 = Root(first_num_ch*2,first_num_ch,3,0)
        self.root_2 = Root(6*first_num_ch,2*first_num_ch,3,0)
        self.root_3 = Root(first_num_ch*8,first_num_ch*4,3,0)
        self.root_4 = Root(first_num_ch*24,first_num_ch*8,3,0)
        
        self.up_4 = UpBlock(16*first_num_ch, 8*first_num_ch,0.00)
        self.up_3 = UpBlock(8*first_num_ch, 4*first_num_ch,0.00)
        self.up_2 = UpBlock(4*first_num_ch, 2*first_num_ch,0.05)
        self.up_1 = UpBlock(2*first_num_ch, first_num_ch,0.1)
        
        self.output = nn.Conv2d(first_num_ch+1, out_num_ch, 1)
        # choose different activation layer
        if output_activation == 'sigmoid':
            self.output_act = nn.Sigmoid()
        elif output_activation == 'tanh':
            self.output_act = nn.Tanh()
        elif output_activation == 'linear':
            self.output_act = nn.Linear()
        else:
            self.output_act = nn.Softplus()
            
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                #nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
                    #nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                #nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                # m.weight.data.normal_(0,0.01)
                m.bias.data.zero_()


    def forward(self, x):
        
        tcn_out = self.tcn(x.squeeze(0)).reshape(-1,120,128).unsqueeze(0)
        
        down_1 = self.down_1(x)
        down_ida_1 = self.down_ida_1(x)
        
        down_2 = self.down_2(self.root_1(down_1,down_ida_1))
       
        down_ida_2 = self.down_ida_2(down_1)
        down_ida_3 = self.down_ida_3(x)
        
        down_3 = self.down_3(self.root_2(down_2,down_ida_2,down_ida_3))
        down_ida_4 = self.down_ida_4(down_2)
        down_4 = self.down_4(self.root_3(down_3,down_ida_4))
        
        down_ida_5 = self.down_ida_5(down_2)
        down_ida_35 = self.down_ida_6(down_3)
        down_ida_15 = self.down_ida_7(x)
        down_5 = self.down_5(self.root_4(down_4,down_ida_5,down_ida_35,down_ida_15))
        
        up_4 = self.up_4(down_4, down_5)
        up_3 = self.up_3(down_3, up_4)
        up_2 = self.up_2(down_2, up_3)
        up_1 = self.up_1(down_1, up_2)
        
        time = self.dc3(self.dc2(self.dc1(F.relu(tcn_out))))
        
        output = self.output(torch.cat([up_1,time],1))
        #output = self.output(up_1)
        output_act = self.output_act(output)

        return output_act
  

if __name__ == "__main__":
    
    X = torch.rand([1,1,120,128])
    #net = DLA_UNet(1,1)
    
    net = DLA_UNet(1,1)
    #net = TCN(128,15360,[i for i in range(1,16)],2,0.2)
    #net.load_state_dict(torch.load(model_path+'best_model_{}_final.path'.format('DLA_UNet_Third'), map_location=torch.device('cpu')))
    result = net(X)
