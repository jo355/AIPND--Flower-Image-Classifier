import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torchvision import datasets,transforms,models
import argparse
import os,json,time
from collections import OrderedDict
#---------------------------------------------------
# Define and parse command-line arguments
#---------------------------------------------------
def get_input_args():
    ''' retrieves command line user inputs via Argparse module.
    Input(s): 
    1.Image folder containing the dataset.
    2.CNN Model Architecture as --arch with default value 'vgg19'.
    3. Text File with Dog Names as --dogfile with default value 'dognames.txt'.
    '''
    parser= argparse.ArgumentParser(description='get command line input')
    parser.add_argument('data_dir',type=str,help='Provide directory')
    parser.add_argument('--save_dir',type=str, help='Provide saving directory. Default is current directory',default= os.getcwd())
    parser.add_argument('--arch',type=str,help='Model architecture: VGG19 or DenseNet121. Enter in lowercase .Default is vgg19.',default='vgg19')
    parser.add_argument('--epochs',type=int,help="Number of epochs. Default= 7",default=7)
    parser.add_argument('--lr',type=float,help='Learning Rate. Default= 0.001',default=0.001)
    parser.add_argument('--hidden',action='append',help= 'Number of hidden units. Default=4096.',type=int ,default= 4096)
    parser.add_argument('--gpu', action='store_true', default=False, help="Enable GPU for training") #mps for mac
    
    return parser.parse_args()
#---------------------------------------------------
# Read and map the flower categories
#---------------------------------------------------
def get_labels():
    '''
    Return(s):
    cat_to_name(dict) - dictionary mapping category labels(values) to their index values(keys).
    '''
    with open('cat_to_name.json','r') as f:
        cat_to_name= json.load(f)
    n_cat= len(cat_to_name.keys())
    print("Categories\n: ",list(set(cat_to_name.values())))
    print("Number of categories: \n",n_cat)  
    return cat_to_name

#---------------------------------------------------
# Loading the datasets and applying transforms
#---------------------------------------------------
def load_transform_data(data_dir):
    '''
    Loads the images from args.data_dir, transforms the datasets.
    Param(s):
    data_dir - input source directory 
    Return(s):
    1. train_loader,test_loader,valid_loader - DataLoaders with transforms applied.
    2.class_to_idx(dict) - dictionary mapping the index (label) to the name of the class.
    '''
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    mean= [0.485,0.456,0.406]
    std= [0.229,0.224,0.225]
    train_transforms=transforms.Compose([transforms.RandomRotation(40),
                                         transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(p=0.3),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean,std) ])
    
    #for valid and test data, only resizing, cropping and normalization done
    valid_test_transforms= transforms.Compose([transforms.Resize(255),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean,std)])
    
    #load datasets with ImageFolder
    train_data= datasets.ImageFolder(train_dir,transform=train_transforms)
    test_data=datasets.ImageFolder(test_dir,transform=valid_test_transforms)
    valid_data=datasets.ImageFolder(valid_dir,transform=valid_test_transforms)
    
    # TODO: Using the image datasets and the trainforms, define the dataloaders 
    train_loader=torch.utils.data.DataLoader(train_data,batch_size= 64, shuffle=True)
    test_loader=torch.utils.data.DataLoader(test_data,batch_size=32)
    valid_loader=torch.utils.data.DataLoader(valid_data,batch_size=32)
    
    return train_loader,valid_loader, test_loader, train_data.class_to_idx
#---------------------------------------------------
# Build and train the model
#---------------------------------------------------
def build_model(arch,hidden,input_size):
    ''' Builds the model.
    Param(s):
    arch(str) - model architecture: VGG19 or DenseNet121 (args.arch).
    hidden(int) - number of hidden units args.hidden.
    Return(s): 
    model- the initialised model.
    '''
    model= getattr(models,arch)(pretrained= True)
    for param in model.parameters():
        param.requires_grad = False
    classifier= nn.Sequential(
        OrderedDict([
            ('fc1',nn.Linear(input_size,hidden)), 
            ('relu1',nn.ReLU()),
            ('dropout1',nn.Dropout(0.05)),
            ('fc2',nn.Linear(hidden,102)),
            ('output',nn.LogSoftmax(dim=1))
        ]))
    # replace the pretrained classifier with the above one.
    model.classifier=classifier
    return model
 

def train_model(train_loader,valid_loader,epochs,lr,arch,hidden,input_size, device):
    '''
    Calls build_model() and trains the network.
    Param(s): 
    trainloader,validloader-  DataLoaders for train and validation datasets.
    epochs- Training epochs.
    lr(float) - learning rate.
    hidden(int) - Number of hidden units.
    device(str) - changes to gpu mode(cuda or mps).
    Return(s): model- Trained and validated network.
    '''
    model= build_model(arch,hidden,input_size)
    
    #set model to gpu mode (cuda or mps)
    model.to(device)
    
    
    
    #set the training hyperparameters
    criterion=nn.NLLLoss()
    optimizer=optim.Adam(model.classifier.parameters(),lr=lr)
    batch=0
    running_loss=running_accuracy= 0
    print_every=20 # frequency of epochs to validate
    training_losses,validation_losses= [],[]
    
    print(f'The device in use is {device}.\n')
    
    #Training the classifier layer
    for e in range(epochs):
        for inputs,labels in train_loader:
            start = time.time()
            batch+=1
            # move inputs and labels to gpu
            inputs,labels= inputs.to(device),labels.to(device)
            #forward pass
            log_ps= model(inputs)
            #calculate loss
            loss= criterion(log_ps,labels)
            #backpropogate
            loss.backward()
            # update weights
            optimizer.step()
            
            #calculate the metrics
            ps= torch.exp(log_ps)
            top_ps,top_class= ps.topk(1,dim=1)
            matches= (top_class==labels.view(*top_class.shape)).type(torch.FloatTensor)
            accuracy= torch.mean(matches)
            
            # resets optimiser gradient and tracks metrics
            optimizer.zero_grad()
            running_loss += loss.item()
            running_accuracy += accuracy.item()
            
            # runs the model on the validation set every 5 loops
            if batch%print_every == 0:
                end = time.time()
                training_time = end-start
                start = time.time()
                # sets the metrics
                validation_loss = 0
                validation_accuracy = 0
                # turns on evaluation mode, gradients turned off
                model.eval()
                with torch.no_grad():
                    for inputs,labels in valid_loader:
                        inputs,labels = inputs.to(device),labels.to(device)
                        log_ps = model(inputs)
                        loss = criterion(log_ps,labels)
                        ps = torch.exp(log_ps)
                        top_ps, top_class = ps.topk(1,dim=1)
                        matches = (top_class == labels.view(*top_class.shape)).type(torch.FloatTensor)
                        accuracy = matches.mean()
                        # tracks validation metrics (test of the model's progress)
                        validation_loss += loss.item()
                        validation_accuracy += accuracy.item()
                # tracks training metrics
                end = time.time()
                validation_time = end-start
                validation_losses.append(running_loss/print_every)
                training_losses.append(validation_loss/len(valid_loader))
                # prints out metrics
                print(f'Epoch {e+1}/{epochs} | Batch {batch}')
                print(f'Running Training Loss: {running_loss/print_every:.3f}')
                print(f'Running Training Accuracy: {running_accuracy/print_every*100:.2f}%')
                print(f'Validation Loss: {validation_loss/len(valid_loader):.3f}')
                print(f'Validation Accuracy: {validation_accuracy/len(valid_loader)*100:.2f}%')

                # resets the metrics and turns on training mode
                running_loss = running_accuracy = 0
                model.train()
    return model
#---------------------------------------------------
# Testing the network
#---------------------------------------------------
def test_model(model,test_loader):
    ''' Evaluates the network on the test dataset.
    Param(s):
    model- trained model.
    test_loader- DataLoader for test data.
    Return(s):
    Test accuracy.
    '''
    #change to cuda for torch.cuda() and mps for backends.mps remember! 
    model.to('cuda') 
    model.eval()
    test_accuracy = 0
    start_time = time.time()
    print('Validation started.')
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to('cuda'), labels.to('cuda')
            log_ps = model(images)
            ps = torch.exp(log_ps)
            top_ps, top_class = ps.topk(1, dim=1)
            matches = (top_class == labels.view(*top_class.shape)).type(torch.FloatTensor)
            accuracy = matches.mean()
            test_accuracy += accuracy.item()
    end_time = time.time()
    print('Validation ended.')
    validation_time = end_time - start_time
    print('Validation time: {:.0f}m {:.0f}s'.format(validation_time / 60, validation_time % 60))
    print(f'Test Accuracy: {test_accuracy / len(test_loader) * 100:.2f}%')

    return test_accuracy / len(test_loader) * 100
#---------------------------------------------------
# Saving the model
#---------------------------------------------------
def save_checkpoint(model, hidden_units, n_output, destination_directory, arch, class_to_idx,input_size):
    ''' Creates checkpoint to save model.
    Param(s):
    model- evaluated model.
    n_output- Number of classes (102) .
    destination_directory- Directory where the model is to be stored.
    arch- Model architecture.
    class_to_idx - dictionary mapping the index (label) to the name of the class. 
    input_size- Number of input neurons.
    '''
    # defines model's checkpoint
    model_checkpoint = {'model_arch':arch, 
                    'clf_input':input_size,
                    'clf_output':n_output,
                    'clf_hidden':hidden_units,
                    'state_dict':model.state_dict(),
                    'model_class_to_index':class_to_idx,
                    }
    
    # saves model in current directory
    if destination_directory:
        torch.save(model_checkpoint,destination_directory+"/"+arch+"_checkpoint.pth")
        print(f"{arch} successfully saved to {destination_directory}")
    else:
        torch.save(model_checkpoint,arch+"_checkpoint.pth")
        print(f"{arch} successfully saved to current directory as {arch}_checkpoint.pth")
        

if __name__ == '__main__':
    
    args = get_input_args()
    data_dir = args.data_dir
    destn_dir = args.save_dir
    hidden_units = args.hidden
    epochs = args.epochs
    arch = args.arch
    lr = args.lr
    gpu=args.gpu

    
    if arch =='vgg19':
        hidden_units=4091
        input_size=25088
    elif arch =='densenet121':
        hidden_units=512
        input_size=1024
    else:
        print("Invalid Architecture .\n")

    print('*** Loading data and defining transforms ...')
    train_loader, valid_loader, test_loader, class_to_idx = load_transform_data(data_dir)
    print('*** Data loaded successfully!\n')
    
    # Label mapping
    print('*** Mapping the category labels to the names ...')
    cat_to_name= get_labels()
    
    #Enabling GPU
    if gpu:
        device = torch.device("cuda") #mps
        print("****** CUDA activated ********")#mps
    else:
        device = torch.device("cpu")
    
    print('*** Model building and training in progress ...')
    print('** Following are training loss, validation loss, and model accuracy:\n')
    model= train_model(train_loader,valid_loader,epochs,lr,arch,hidden_units,input_size, device)
    
    print('*** Testing in progress ...')
    test_accuracy= test_model(model,test_loader)
    
    print(f"*** Saving {arch}'s checkpoint ...")
    save_checkpoint(model,hidden_units,102,destn_dir,arch,class_to_idx,input_size)
    print("*** Saved checkpoint!...")
    
    print("*** Training successfully completed ...")
    
    

