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
    #df = np.vstack((imgs_pixel[0].values,imgs_pixel[1].values,imgs_pixel[2].values,imgs_pixel[3].values,imgs_pixel[4].values,imgs_pixel[5].values,imgs_pixel[6].values,imgs_pixel[7].values,imgs_pixel[8].values,imgs_pixel[9].values,imgs_pixel[10].values,imgs_pixel[11].values,imgs_pixel[12].values,imgs_pixel[13].values,imgs_pixel[14].values,imgs_pixel[15].values,imgs_pixel[16].values,imgs_pixel[17].values,imgs_pixel[18].values,imgs_pixel[19].values,imgs_pixel[20].values,imgs_pixel[21].values,imgs_pixel[22].values,imgs_pixel[23].values,imgs_pixel[24].values,imgs_pixel[25].values,imgs_pixel[26].values,imgs_pixel[27].values))
        
    plt.imsave('1_new.png', np.array(imgs_pixel).reshape(28,28))
    
        
    

def main():
    rows = load_image_binary_data("./data/train.csv")
    #img = convert_pixel_to_image(rows.iloc[:1,:])    
    display_image(rows.iloc[:1,:])



if __name__ == "__main__":
    main()
