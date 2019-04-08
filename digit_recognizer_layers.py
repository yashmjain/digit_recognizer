# -*- coding: utf-8 -*-

import numpy
import torch
import torch.nn as nn

class cnn(nn.Module):
    def __init__(self,in_channel,num_filter,kernel_size,padding,stride):
        super(cnn, self).__init__()
        self.cnn = nn.Conv2d(in_channels=in_channel, out_channels=num_filter, kernel_size=kernel_size,stride=1,
                             padding=0)        
        
        
    def forward(self,input):
        self.cnn(input)
        
    

