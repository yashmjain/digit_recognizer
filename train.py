# -*- coding: utf-8 -*-

from digit_recognizer_model import digit_recognizer,OUT_CHANNEL
from pre_processing import display_image,load_image_binary_data
import numpy as np
import torch


def main():
    IMAGE_TRAIN_PIXEL_FILE = "./data/train.csv"
    print ()
    dr = digit_recognizer()
    #Process the data before giving input to the model
    images_data = load_image_binary_data(IMAGE_TRAIN_PIXEL_FILE)
    load_first_image = images_data.iloc[:1,:]
    load_first_image = np.array(load_first_image).reshape(28,28)
    print("Display original image")
    display_image(load_first_image)
    load_first_image = torch.from_numpy(load_first_image).type(torch.FloatTensor)
    load_first_image = load_first_image.unsqueeze(0).unsqueeze(0)
    output = dr(load_first_image)
    for filter_no in range(OUT_CHANNEL):       
        display_image(output[0][filter_no].detach().numpy())
    

if __name__ == '__main__':
    main()

