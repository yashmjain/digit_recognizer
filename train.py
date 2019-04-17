# -*- coding: utf-8 -*-

from digit_recognizer_model import digit_recognizer,OUT_CHANNEL
from pre_processing import display_image,load_image_binary_data
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
#import util



def main():
    IMAGE_TRAIN_PIXEL_FILE = "./data/train.csv"
    print ()
    dr = digit_recognizer()
    #Process the data before giving input to the model
    images_data,label_data = load_image_binary_data(IMAGE_TRAIN_PIXEL_FILE)
    load_first_image = images_data.iloc[:1,:]
    load_first_image = np.array(load_first_image).reshape(28,28)
    pdf = PdfPages('./data/first_image_pass_first_cnnlayer_with_relu_maxpool_outchannel_5_with_topmost_original_image.pdf')
    print("Display original image")
    display_image(load_first_image,pdf)
    load_first_image = torch.from_numpy(load_first_image).type(torch.FloatTensor)
    load_first_image = load_first_image.unsqueeze(0).unsqueeze(0)
    #output = dr(load_first_image)
    
    # Display the first image after passign through one cnn layer with relu and maxpool
#    for filter_no in range(OUT_CHANNEL):       
#        display_image(output[0][filter_no].detach().numpy(),pdf)
#    pdf.close()
    
    
    # Divide the train data into train data and dev eval data 
    # It seems adding 1 as an extra dimension does not have any impact 
    images_data =  np.array(images_data).reshape(-1,1,28,28)
    label_data = np.array(label_data)
    x_train, x_valid, y_train, y_valid = train_test_split(images_data, label_data, test_size=0.2)
    print("The shape of training data: ",x_train.shape)
    print("The shape of testing  data: ",x_valid.shape)
    print("The shape of training label data: ",y_train.shape)
    print("The shape of testing  label data: ",y_valid.shape)

    #Initialize the model 
    model = digit_recognizer()
    
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_train).float(),
                                       torch.from_numpy(y_train).long())
    
    eval_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_valid).float(),
                                      torch.from_numpy(y_valid).long())
    
    #create train dataloader object 
    train_loader = data.DataLoader(dataset=train_dataset,batch_size=150,shuffle=True)   
    
    #create eval dataloader object
    eval_loader = data.DataLoader(dataset=eval_dataset,batch_size=150,shuffle=False)   

    
    #Initialize the optimizer 
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    #initialize the cross entropy loss 
    loss_fn = nn.CrossEntropyLoss()
    
    #Declare the checkpointsaver to save the model
    
#    saver = util.CheckpointSaver('./trained_model',
#                                 max_checkpoints=4,
#                                 metric_name='NLL',
#                                 maximize_metric=args.maximize_metric,
#                                 log=log)
    
    epoch = 30 
    
    for i in range(epoch):
        train(train_loader,optimizer,model,loss_fn)        
        print("Starting evaluation ")
        evaluate(eval_loader,model,loss_fn)
        print("Completed epoch ",i)
        
    
    
    
    
    
    
    
def train(train_loader,optimizer,model,loss_fn):    
    #train the model   
    avg_loss = 0
    cum_loss = 0
    running_corrects = 0
    for batch_no,(train_image_data,train_label_data) in enumerate(train_loader,start =1 ):
        optimizer.zero_grad()
        output = model(train_image_data)       #train_image_data has shape batch_size*in_channel*H*W
        loss = loss_fn(output, train_label_data.squeeze()) 
        print("The loss during training is  :: ",loss.item())
        cum_loss = cum_loss + loss.item()
        avg_loss = cum_loss/batch_no
        print("The average loss across batch is :: ",avg_loss)
        
        #Calculate the accuracy 
        #actual_label = torch.argmax()
        
        #accuracy = torch.sum(preds == train_label_data)
        #accuracy_epoch = accuracy_epoch +accuracy.item()        
     
        
        print("The batch no is ",batch_no)
        loss.backward()
        optimizer.step()
        
def evaluate(eval_loader,model,loss_fn):
    #eval the data
    cum_loss = 0
    avg_eval_loss = 0
    running_corrects = 0 
    avg_accuracy = 0
    for batch_no,(eval_image_data,eval_label_data) in enumerate(eval_loader,start = 1):
        eval_output = model(eval_image_data)
        eval_loss = loss_fn(eval_output, eval_label_data.squeeze())
        print("The loss during eval_loss is  :: ",eval_loss.item())
        cum_loss = cum_loss + eval_loss.item()
        avg_eval_loss = cum_loss/batch_no
        print("The average evaluation loss across batch is :: ",avg_eval_loss)
        preds = torch.argmax(eval_output,dim=1)
        running_corrects = (torch.sum(preds == eval_label_data.squeeze()).item()/(len(eval_label_data)))
        print("The accuracy across epoch is ::",running_corrects)
        
        avg_accuracy =  avg_accuracy + running_corrects
        print("The average accuracy is :: ",avg_accuracy/batch_no)

        
    

if __name__ == '__main__':
    main()

