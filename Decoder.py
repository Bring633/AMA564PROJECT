#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 10:40:49 2022

@author: bring
"""

import torch
from torch import nn
import torch.nn.functional as F

    
class Channel_reduce(nn.Module):
    
    #对应decoder的绿线
    
    def __init__(self,in_channels,out_channels,):
        
        super(Channel_reduce,self).__init__()
        
        self.maxpool = nn.MaxPool2d(3,stride=1,padding=1)
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1)
        
        self.conv2 = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1)
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


class Conv_block(torch.nn.Module):
    
    def __init__(self,in_channels,out_channels):
        
        super(Conv_block,self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels,out_channels, stride = 1,padding=1,kernel_size=3)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        return None
    
    def forward(self,x):
        
        return F.relu(self.bn1(self.conv1(x)))

class Block2(nn.Module):
    
    def __init__(self,in_channels,out_channels,skip_outchannels):
        
    #inchannels就是输入的channel，skip 就是 agg 模块输入的channel，一般是aggchannel/2
        
        super(Block2,self).__init__()
        
        self.conv1 = Conv_block(in_channels,out_channels)
        self.conv2 = Conv_block(out_channels,out_channels)
        
        self.conv3 = Conv_block(out_channels+skip_outchannels,out_channels)
        
        return None
    
    def forward(self,x,*x_pass):

        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        
        cat_tensor = torch.cat(x_pass,1) 

        out = self.conv3(torch.cat((cat_tensor,conv2_out),1))

        return out

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

class Level1(nn.Module):
    
    def __init__(self,in_channels,out_channels,num_input):
    #inchannels是单个输入的channel
        
        super(Level1,self).__init__()
    
        self.root = Root(in_channels*num_input,in_channels*num_input,3,False)
        self.conv1 = Conv_block(in_channels,in_channels)
        self.block2 = Block2(in_channels,out_channels,in_channels)
        
        self.reduce1 = Channel_reduce(in_channels*num_input,in_channels)#conv block
        self.reduce2 = Channel_reduce(in_channels*num_input,in_channels)#bl2 skip
        
        return None
    
    def forward(self,*x):
        
        root_out = self.root(*x)
        
        conv_out = self.conv1(self.reduce1(root_out))
        blk_out = self.block2(conv_out,self.reduce2(root_out))
        
        return blk_out

class Level2(nn.Module):
    
    def __init__(self,in_channels,out_channels,num_input):
        
        super(Level2,self).__init__()
        self.root = Root(in_channels*num_input,in_channels*num_input,3,False)
        self.conv1 = Conv_block(in_channels,in_channels)
        self.bl2 = Block2(in_channels,in_channels//2,(in_channels*num_input)//2)
        self.level1 = Level1(in_channels//2,out_channels,2)
        
        self.reduce1 = Channel_reduce(in_channels*num_input,in_channels)
        self.reduce2 = Channel_reduce(in_channels*num_input,(in_channels*num_input)//2)
        self.reduce3 = Channel_reduce(in_channels*num_input,in_channels//2)

        return None
    
    def forward(self,*x):
        
        root_out = self.root(*x)
        
        conv_out = self.conv1(self.reduce1(root_out))
        bl2_out = self.bl2(conv_out,self.reduce2(root_out))
        
        level1_out = self.level1(self.reduce3(root_out),bl2_out)
        
        return level1_out,root_out
    
class DoubleConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size = 3,stride = 1),
                torch.nn.BatchNorm2d(out_channels),  #BN层
                torch.nn.ReLU(),
                torch.nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1,stride = 1),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(),
                #torch.nn.MaxPool2d(2)
                )
        
    def forward(self, x):
        return self.double_conv(x)
    
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Upsample,self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
            
        return None

    def forward(self, x):
        
        x = self.up(x)
        x = F.pad(x,(3//2,int(3/2),3//2,int(3/2)))
        x =self.conv(x)
            
        return x
    

class Decoder(nn.Module):
    
    def __init__(self,):
        
        super(Decoder,self).__init__()
        
        self.conv1 = Conv_block(4928,2688)
        self.bl2_1 = Block2(2688,1344,4928)
        self.reduce_conv1 = Channel_reduce(9856,4928)
        self.reduce_conv2 = Channel_reduce(9856,4928)
        
        
        self.level1_1 = Level1(1344,448,2)
        self.reduce_level1_1 = Channel_reduce(9856,1344)
        
        self.reduce_level2_1 = Channel_reduce(9856,448)
        self.level2_1 = Level2(448,192,3)
        
        self.reduceida_1 = Channel_reduce(9856,448)
        self.up1 = Upsample(448,192)
        self.up2 = Upsample(192,192)
        
        self.level2_2 = Level2(192,32,3)
        
        self.reduce2 = Channel_reduce(576,96)
        self.up3 = Upsample(96,32)
        self.up4 = Upsample(32,32)
        
        self.level1_2 = Level1(32,32,2)
        
        self._1x1conv = torch.nn.Conv2d(32,1,kernel_size = 1)
        
        return None
    
    def forward(self,x,encoder_agg1,encoder_agg2):
        
        conv_in = self.reduce_conv1(x)#[1, 4928, 30, 32])
        conv1_out = self.conv1(conv_in)#([1, 2688, 30, 32])
        bl2_1_out = self.bl2_1(conv1_out,self.reduce_conv2(x))#([1, 1344, 30, 32])
        
        level1_out = self.level1_1(bl2_1_out,self.reduce_level1_1(x))
        
        level2_1_out,not_use = self.level2_1(level1_out,self.reduce_level2_1(x),encoder_agg2)
        
        level2_2_in1 = self.up1(self.reduceida_1(x))
        level2_2_in2 = self.up2(level2_1_out)
        
        level2_2_out,level2_2_ida = self.level2_2(level2_2_in2,level2_2_in1,encoder_agg1)
        
        level1_2_in1 = self.up3(self.reduce2(level2_2_ida))
        level1_2_in2 = self.up4(level2_2_out)
        
        level1_2_out = self.level1_2(level1_2_in2,level1_2_in1)
        
        out = self._1x1conv(level1_2_out)
        
        return out
    
class DecoderSmall(nn.Module):
    
    def __init__(self,):
        
        super(DecoderSmall,self).__init__()
        
        self.conv1 = Conv_block(1232,616)
        self.bl2_1 = Block2(616,336,1232)
        self.reduce_conv1 = Channel_reduce(2464,1232)
        self.reduce_conv2 = Channel_reduce(2464,1232)
        
        
        self.level1_1 = Level1(336,336,2)#?
        self.reduce_level1_1 = Channel_reduce(2464,336)
        
        self.reduce_level2_1 = Channel_reduce(2464,336)
        self.level2_1 = Level2(336,48,3)
        
        self.reduceida_1 = Channel_reduce(2464,336)
        self.up1 = Upsample(336,48)
        self.up2 = Upsample(48,48)
        
        self.level2_2 = Level2(48,8,3)
        
        self.reduce2 = Channel_reduce(144,24)
        self.up3 = Upsample(24,8)
        self.up4 = Upsample(8,8)
        
        self.level1_2 = Level1(8,8,2)
        
        self._1x1conv = torch.nn.Conv2d(8,1,kernel_size = 1)
        
        return None
    
    def forward(self,x,encoder_agg1,encoder_agg2):
        
        conv_in = self.reduce_conv1(x)#[1, 4928, 30, 32])
        conv1_out = self.conv1(conv_in)#([1, 2688, 30, 32])
        bl2_1_out = self.bl2_1(conv1_out,self.reduce_conv2(x))#([1, 1344, 30, 32])
        
        level1_out = self.level1_1(bl2_1_out,self.reduce_level1_1(x))
        
        level2_1_out,not_use = self.level2_1(level1_out,self.reduce_level2_1(x),encoder_agg2)
        
        level2_2_in1 = self.up1(self.reduceida_1(x))
        level2_2_in2 = self.up2(level2_1_out)
        
        level2_2_out,level2_2_ida = self.level2_2(level2_2_in2,level2_2_in1,encoder_agg1)
        
        level1_2_in1 = self.up3(self.reduce2(level2_2_ida))
        level1_2_in2 = self.up4(level2_2_out)
        
        level1_2_out = self.level1_2(level1_2_in2,level1_2_in1)
        
        out = self._1x1conv(level1_2_out)
        
        return out
    
class DecoderMini(nn.Module):
    
    def __init__(self,):
        
        super(DecoderMini,self).__init__()
        
        self.level2_2 = Level2(48,8,1)
        
        self.reduce2 = Channel_reduce(48,24)
        self.up3 = Upsample(24,8)
        self.up4 = Upsample(8,8)
        
        self.level1_2 = Level1(8,8,2)
        
        self._1x1conv = torch.nn.Conv2d(8,1,kernel_size = 1)
        
        return None
    
    def forward(self,x):
        
        level2_2_out,level2_2_ida = self.level2_2(x)
        
        level1_2_in1 = self.up3(self.reduce2(level2_2_ida))
        level1_2_in2 = self.up4(level2_2_out)
        
        level1_2_out = self.level1_2(level1_2_in2,level1_2_in1)
        
        out = self._1x1conv(level1_2_out)
        
        return out
    
class DecoderMini2(nn.Module):
    
    def __init__(self,):
        
        super(DecoderMini2,self).__init__()
        
        self.level2_1 = Level2(336,48,1)

        self.up1 = Upsample(336,48)
        self.up2 = Upsample(48,48)
        
        self.level2_2 = Level2(48,8,2)
        
        self.reduce2 = Channel_reduce(96,24)
        self.up3 = Upsample(24,8)
        self.up4 = Upsample(8,8)
        
        self.level1_2 = Level1(8,8,2)
        
        self._1x1conv = torch.nn.Conv2d(8,1,kernel_size = 1)
        
        return None
    
    def forward(self,encoder_agg1,encoder_agg2):
        
        level2_1_out,not_use = self.level2_1(encoder_agg2)

        level2_2_in2 = self.up2(level2_1_out)
        
        level2_2_out,level2_2_ida = self.level2_2(level2_2_in2,encoder_agg1)
        
        level1_2_in1 = self.up3(self.reduce2(level2_2_ida))
        level1_2_in2 = self.up4(level2_2_out)
        
        level1_2_out = self.level1_2(level1_2_in2,level1_2_in1)
        
        out = self._1x1conv(level1_2_out)
        
        return out


if __name__ == "__main__":
    
    decoder = DecoderMini2()
    #encoder_out = torch.rand((1, 2464, 30, 32))
    encoder_agg1 = torch.rand([1, 48, 60, 64])
    encoder_agg2 = torch.rand([1, 336, 30, 32])
    #decoder_out = decoder(encoder_out,encoder_agg1,encoder_agg2)
    decoder_out = decoder(encoder_agg1,encoder_agg2)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    