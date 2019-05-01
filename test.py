# -*- coding: utf-8 -*-

import pandas as pd
import sys
from pre_processing import display_image,load_image_binary_data
import torch
import torch.utils.data as data
import util
import numpy as np
from digit_recognizer_model import digit_recognizer

def main(args):
    READ_TEST_CSV = './data/test.csv'
    checkpoint_path = args[1]
    test_image_data = read_test_data(READ_TEST_CSV)
    test_image_data = np.array(test_image_data).reshape(-1,1,28,28)
    test_image_data = torch.from_numpy(test_image_data).float()
    
    model = load_model(checkpoint_path)
    test_output = test(test_image_data,model)  
    write_csv(test_output)
    
    
def load_model(checkpoint_path):
    model = digit_recognizer()     
    device,gpu_ids = util.get_available_devices()
    model = util.load_model(model,checkpoint_path,gpu_ids,False)
    model = model.to(device)
    return model
    
    
    
def read_test_data(path_test_csv):
    return pd.read_csv(path_test_csv)


def test(test_image_data,model):
    #test the data
    test_output = model(test_image_data) # test image should be of shape N * Cin * W * H (N : batch_size , Cin : In channels , W : Width , H : Height)
    test_output = torch.argmax(test_output,dim=1)
    return test_output
    
    
def write_csv(test_output):        
    df = pd.DataFrame(test_output.numpy(),columns=['Label'],dtype=np.int8)
    df.index +=1
    df.to_csv('./test/submission.csv',index_label='ImageId')
    
        
    

if __name__ == '__main__':
    main(sys.argv[:])