# -*- coding: utf-8

from pre_processing import load_image_binary_data,display_image
import digit_recognizer_layers
import torch
import torch.nn as nn
import numpy as np

IMAGE_TRAIN_PIXEL_FILE = "./data/train.csv"
IN_CHANNEL = 1 
OUT_CHANNEL = 2
KERNEL_SIZE = 3
PADDING = 1

class digit_recognizer(nn.Module):
    def __init__(self):
        super(digit_recognizer, self).__init__()
        self.cnn = digit_recognizer_layers.cnn(in_channel=IN_CHANNEL,num_filter=OUT_CHANNEL,kernel_size=KERNEL_SIZE,padding=PADDING)
        
        
    def forward(self):
        images_data = load_image_binary_data(IMAGE_TRAIN_PIXEL_FILE)
        load_first_image = images_data.iloc[:1,:]
        load_first_image = np.array(load_first_image).reshape(28,28)
        load_first_image = torch.from_numpy(load_first_image).type(torch.FloatTensor)
        load_first_image = load_first_image.unsqueeze(0).unsqueeze(0)
        output_image = self.cnn(load_first_image)
        return output_image
        

        
    
    
    