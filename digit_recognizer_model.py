# -*- coding: utf-8

from pre_processing import load_image_binary_data,display_image
import digit_recognizer_layers
import torch
import torch.nn as nn
import numpy as np

IMAGE_TRAIN_PIXEL_FILE = "./data/train.csv"
IN_CHANNEL = 1 
OUT_CHANNEL = 5
KERNEL_SIZE = 3
PADDING = 1

class digit_recognizer(nn.Module):
    def __init__(self):
        super(digit_recognizer, self).__init__()
        self.cnn = digit_recognizer_layers.cnn(in_channel=IN_CHANNEL,num_filter=OUT_CHANNEL,kernel_size=KERNEL_SIZE,padding=PADDING)
        
        
    def forward(self,input):       
        output_image = self.cnn(input)  # input shape batch_size,out_channels,Width,Height
        return output_image
        

        
    
    
    