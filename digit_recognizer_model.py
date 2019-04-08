# -*- coding: utf-8

from pre_processing import load_image_binary_data
import digit_recognizer_layers

IMAGE_TRAIN_PIXEL_FILE = "./data/train.csv"

class digit_recognizer():
    images_data = load_image_binary_data(IMAGE_TRAIN_PIXEL_FILE)
    digit_recognizer_layers.cnn(in_channel=3,out_channels=1,kernel_size=3,padding=2)
    
    
    