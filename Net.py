#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 14:12:23 2022

@author: bring
"""

from torch import nn
import torch
from Decoder import Decoder,Channel_reduce,DecoderSmall,DecoderMini,DecoderMini2
from Encoder import Encoder,EncoderSmall,EncoderMini,EncoderMini2
import torch.nn.functional as F

class Network(nn.Module):
    
    def __init__(self,):
        
        super(Network,self).__init__()
        
        self.reduce1 = Channel_reduce(448,192)
        self.reduce2 = Channel_reduce(1344,448)
        
        self.encoder = Encoder()
        self.decoder = Decoder()
        
        return None 
        
        
    def forward(self,x):
        
        x_first = x/255
        encoder_out,encoder_agg1,encoder_agg2  = self.encoder(x)#(1, 9856, 30, 32),([1, 448, 60, 64]),([1, 1344, 30, 32]))
        decoder_out = self.decoder(encoder_out,self.reduce1(encoder_agg1),self.reduce2(encoder_agg2))
        
        x = decoder_out+x_first
        
        return x
    
class NetworkSmall(nn.Module):
    
    def __init__(self,):
        
        super(NetworkSmall,self).__init__()
        
        self.reduce1 = Channel_reduce(112,48)
        self.reduce2 = Channel_reduce(336,336)
        
        self.encoder = EncoderSmall()
        self.decoder = DecoderSmall()
        
        return None 
        
        
    def forward(self,x):
        
        x_first = x/255
        encoder_out,encoder_agg1,encoder_agg2  = self.encoder(x)#(1, 9856, 30, 32),([1, 448, 60, 64]),([1, 1344, 30, 32]))
        decoder_out = self.decoder(encoder_out,self.reduce1(encoder_agg1),self.reduce2(encoder_agg2))
        
        x = decoder_out+x_first
        
        return x

class NetworkMini(nn.Module):
    
    def __init__(self,):
        
        super(NetworkMini,self).__init__()
        
        self.reduce1 = Channel_reduce(112,48)
        
        self.encoder = EncoderMini()
        self.decoder = DecoderMini()
        
        return None 
        
        
    def forward(self,x):
        
        x_first = x
        encoder_agg1  = self.encoder(x)#(1, 9856, 30, 32),([1, 448, 60, 64]),([1, 1344, 30, 32]))
        decoder_out = self.decoder(self.reduce1(encoder_agg1))
        
        x = decoder_out+x_first
        
        return x
    
class NetworkMini2(nn.Module):
    
    def __init__(self,):
        
        super(NetworkMini2,self).__init__()
        
        self.reduce1 = Channel_reduce(112,48)
        self.reduce2 = Channel_reduce(336,336)
        
        self.encoder = EncoderMini2()
        self.decoder = DecoderMini2()
        
        return None 
        
        
    def forward(self,x):
        
        x_first = x
        encoder_agg1,encoder_agg2  = self.encoder(x)#(1, 9856, 30, 32),([1, 448, 60, 64]),([1, 1344, 30, 32]))
        decoder_out = self.decoder(self.reduce1(encoder_agg1),self.reduce2(encoder_agg2))
        
        x = decoder_out+x_first
        
        return x

class ConvDoubleBlock1(nn.Module):
    def __init__(self, in_num_ch, out_num_ch,dropout, filter_size=3):
        super(ConvDoubleBlock1, self).__init__()
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

class UpBlock(nn.Module):
    def __init__(self, in_num_ch, out_num_ch,cat=True, dropout=0,upsample=True):
        super(UpBlock, self).__init__()
        if upsample == True:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_num_ch, in_num_ch//2, 2)
                )
        else:
            self.up = nn.ConvTranspose2d(in_num_ch, in_num_ch//2, 3, padding=1, stride=2) # (H-1)*stride-2*padding+kernel_size
        #1
        
        self.conv = ConvDoubleBlock1(in_num_ch, out_num_ch, dropout,3)
        if cat==False:
            self.conv = ConvDoubleBlock1(in_num_ch//2, out_num_ch, dropout,3)

    def forward(self, x_down, x_up,cat=True):
        
        x_up = self.up(x_up)
        diffY = (x_down.size()[2] - x_up.size()[2])
        diffX = (x_down.size()[3] - x_up.size()[3])
        x_up = F.pad(x_up, (diffX // 2, diffX - diffX//2, diffY // 2, diffY - diffY//2))  
        if cat:
            x = torch.cat([x_up, x_down], dim=1)
        else:
            x = x_up
        x = self.conv(x)
    
        """
        x_up = self.up(x_up)
        # after the upsample/convtrans, the HW is smaller than the down-sampling map
        x_up = F.pad(x_up, (3//2,int(3/2),3//2,int(3/2)), mode='replicate')
        x = torch.cat([x_down, x_up], 1)
        x = self.conv(x)
        """
        return x

class ConvDoubleBlock(nn.Module):
    def __init__(self, in_num_ch, out_num_ch,dropout=0,stride=2, filter_size=3):
        super(ConvDoubleBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_num_ch, out_num_ch, filter_size, padding=1,stride=1),
            nn.BatchNorm2d(out_num_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(out_num_ch, out_num_ch, filter_size, padding=1,stride=stride),
            nn.BatchNorm2d(out_num_ch),
            nn.ReLU(inplace=True),
            #nn.Dropout(dropout),
            )

    def forward(self, x):
        x = self.conv(x)
        return x

class REDCNNNet(nn.Module):
    def __init__(self):
        super(REDCNNNet,self).__init__()
        self.conv1 = ConvDoubleBlock(1,32)
        self.conv2 = ConvDoubleBlock(32,64)
        self.conv3 = ConvDoubleBlock(64,128)
        self.conv4 = ConvDoubleBlock(128,256,stride=1)
        
        self.invconv5 = UpBlock(256, 128)
        self.invconv6 = UpBlock(128, 64)
        self.invconv7 = UpBlock(64, 32)
        self.invconv8 = UpBlock(32, 1,False)
        self.conv9 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0, bias=True),
             nn.ReLU()
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0, bias=True)
        )
        
    def forward(self, x):
#        encode
        resi_dual_1 = x
        out = self.conv1(x)
        resi_dual_2 = out
        out = self.conv2(out)
        resi_dual_3 = out
        out = self.conv3(out)
        resi_dual_4 = out
        out = self.conv4(out)   
        
        out = self.invconv5(resi_dual_4,out)
        #out = out+resi_dual_4
        
#        decode
        out = self.invconv6(resi_dual_3,out)
        #out = out+resi_dual_3
        out = self.invconv7(resi_dual_2,out)
        #out = out+resi_dual_2
        out = self.invconv8(resi_dual_1,out,False)

        #out = out+resi_dual_1
        out = self.conv9(out)
        
        return out    



if __name__ == "__main__":
    
    
    net = REDCNNNet()
    vec = torch.rand([1,1,120,128])
    result = net(vec)
