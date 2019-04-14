# -*- coding: utf-8

from pre_processing import load_image_binary_data,display_image
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

IMAGE_TRAIN_PIXEL_FILE = "./data/train.csv"
IN_CHANNEL = 1 
OUT_CHANNEL = 5
KERNEL_SIZE = 3
PADDING = 1

X_channel = 1
filter1, kernel1, padding1, max_pooling1 = 16, 3, 1, 2
filter2, kernel2, padding2, max_pooling2 = 32, 3, 1, 2
filter3, kernel3, padding3, max_pooling3 = 64, 3, 1, 2
dense0, dense1, dense2, dense3 = 64*3*3, 120, 64, 10

class digit_recognizer(nn.Module):
    def __init__(self):
        super(digit_recognizer, self).__init__()
        self.conv1 = nn.Conv2d(X_channel, filter1, kernel1, padding=padding1) 
        self.conv2 = nn.Conv2d(filter1, filter2, kernel2, padding=padding2) 
        self.conv3 = nn.Conv2d(filter2, filter3, kernel3, padding=padding3) 
        
        # fully connect
        self.fc1 = nn.Linear(dense0, dense1)
        self.fc2 = nn.Linear(dense1, dense2)
        self.fc3 = nn.Linear(dense2, dense3)
        
        
    def forward(self,X):       
        X = F.max_pool2d(F.relu(self.conv1(X)), max_pooling1)
        X = F.max_pool2d(F.relu(self.conv2(X)), max_pooling2)
        X = F.max_pool2d(F.relu(self.conv3(X)), max_pooling3)
        X = X.view(-1,dense0)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return X

        
    
    
    