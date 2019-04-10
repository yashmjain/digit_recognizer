# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def load_image_binary_data(file):
    num_of_columns = 785 #This include label column also
    return pd.read_csv(file,usecols = [i for i in range(num_of_columns) if i != 0])   

def convert_pixel_to_image(img_pixel):
    img =[]
    col=0
    row=0
    while (col<784):
        print("The value of i is ",col ," and the value of i+28 is ",col+28)
        img.append(img_pixel.iloc[:,col:col+28])
        row=row+1
        col=col+28
    return img

def display_image(imgs_pixel):   
    #print("The pixel of the image is ::",imgs_pixel)
    plt.imshow(imgs_pixel,cmap='gray')
    plt.show()
        
    

def main():
    rows = load_image_binary_data("./data/train.csv")
    #img = convert_pixel_to_image(rows.iloc[:1,:])    
    display_image(np.array(rows.iloc[:1,:]).reshape(28,28))



if __name__ == "__main__":
    main()
