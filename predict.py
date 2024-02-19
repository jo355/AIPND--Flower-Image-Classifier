import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import time,json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import seaborn as sb

def get_predict_args():
    """
    Parses command-line arguments.
    Return(s):
    args (argparse.Namespace)- Parsed command-line arguments.
    """
    parser= argparse.ArgumentParser(description='get command line input for prediction')
    parser.add_argument('image_path', type=str, help='Path to the image to be classified')
    parser.add_argument('checkpoint', type=str, help='Path to the checkpoint file')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top predictions to return')
    parser.add_argument('--cat_names', type=str, default='cat_to_name.json', help='Path to the JSON file with category names')
    parser.add_argument('--gpu', action='store_true', help='Enable GPU for inference',default= True)
    args = parser.parse_args()
    return args



def load_checkpoint_model(checkpoint_path,device):
    ''' 
    Loads the model at the checkpoint_path.
    Param(s):
    device - switch between gpu and cpu.
    checkpoint_path - saved model's path.
    Return(s): 
    Loaded model.
    '''
    # Define map_location based on the specified device
    map_location = 'cuda' if device == 'gpu' and torch.cuda.is_available() else 'cpu'
    #map_location ='mps' if device =='gpu' and torch.backends.mps.is_available() else 'cpu'
    checkpoint = torch.load(checkpoint_path,map_location)
    if checkpoint['model_arch'] == 'vgg19':
        model = models.vgg19(pretrained=True)
        input_features = 25088
    elif checkpoint['model_arch'] == 'densenet121':
        model = models.densenet121(pretrained=True)
        input_features = 1024
    else:
        print('Error: Unrecognized architecture')
    for param in model.parameters():
            param.requires_grad = False

    model.class_to_idx = checkpoint['model_class_to_index']
    hidden_units = checkpoint['clf_hidden']
    output_units = checkpoint['clf_output']
    
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_features, hidden_units)),
        ('ReLu1', nn.ReLU()),
        ('Dropout1', nn.Dropout(p=0.05)),
        ('fc2', nn.Linear(hidden_units, output_units)),
        ('output', nn.LogSoftmax(dim=1))
    ]))    
    
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array.
    '''
    
    pil_image=Image.open(image).convert('RGB')
    #resize and center crop
    pil_image=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])(pil_image)
    #convert tensor to numpy array
    processed_image=pil_image.numpy()
    return processed_image

def class_to_label(file, classes):
    '''
    Maps predicted classes(labels) to the respective category names.
    Param(s):
    file- contains the index to name mapping of flower categories.
    classes- predicted classes/ labels.
    '''
    with open(file, 'r') as f:
        class_mapping = json.load(f)
    return [class_mapping[c] for c in classes]


def predict(image_path, model, idx_mapping, topk, device='cpu'):
    '''
    Generates the model's probability predictions for top k classes.
    Param(s):
    topk (int) - number of top classes predictions' for test input
    model - loaded model
    idx_mapping - maps classes' indexes to their labels(integer form)
    Return(s):
    list_ps(list) - probabilities of the top k classes.
    classes - indexes of the classes.
    '''
    model.to(device).eval()
    
    pre_processed_image = torch.unsqueeze(torch.from_numpy(process_image(image_path)), 0).to(device).float()
    
    with torch.no_grad():
        log_ps = model(pre_processed_image)
    
    ps = torch.exp(log_ps)
    top_ps, top_idx = ps.topk(topk, dim=1)
    
    list_ps = top_ps.tolist()[0]
    list_idx = top_idx.tolist()[0]
    classes= []
    model.train()
    classes = [idx_mapping[x] for x in list_idx]
    return list_ps, classes


def print_preds(probabilities, classes, image, category_names=None):
    
    print(image)
    
    if category_names:
        labels = class_to_label(category_names, classes)
        for i, (ps, ls, cs) in enumerate(zip(probabilities, labels, classes), 1):
            print(f'{i}) {ps * 100:.2f}% {ls.title()} | Class No. {cs}')
    else:
        for i, (ps, cs) in enumerate(zip(probabilities, classes), 1):
            print(f'{i}) {ps * 100:.2f}% Class No. {cs} ')
    
    print('')

if __name__ == '__main__':
    
    args = get_predict_args()
    loaded_model = load_checkpoint_model(args.checkpoint,device='cpu')
    idx_mapping = {v: k for k, v in loaded_model.class_to_idx.items()}
    probabilities, classes = predict(args.image_path, loaded_model,idx_mapping, topk=args.top_k, device="cpu")
    print_preds(probabilities, classes, args.image_path, 'cat_to_name.json')
    




    