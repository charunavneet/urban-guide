import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from torch import tensor
from torch import optim
from torchvision import datasets,transforms
import PIL
from collections import OrderedDict
import argparse
from PIL import Image
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import json
from train import train 
%matplotlib inline


parser = argparse.ArgumentParser()
parser.add_argument('--image',help='points to image files',required=True,type=str)
parser.add_argument('--checkpoint',help='points to check point files',required=True,type=str)
parser.add_argument('--k_top',dest='ktop',action="store",default=5,help='select top k matches ')
parser.add_argument('--category_names',dest='category_names',action='store',default='cat_to_name.json')
parser.add_argument('--gpu',dest='gpu',action='store',default=False)
argp=parser.parse_args()

def load_checkpoint(path):
    checkpoint = torch.load('checkpoint.pth')
    model = model.vgg19(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
        
    model.class_to_idx = checkpoint[class_to_idx]
    model.load_state_dict(checkpoint['state_dict'])
    return model
def process_image(image):
    pil_image = Image.open(image)
    
    process_image = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                                ])
    np_image = process_image(pil_image)
    return np_image
def main():
    model = load_checkpoint('model_checkpoint.pth')
    if argp.gpu and torch.cuda.ias_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    model.to(device)
    image = process_image(image).to(device)
    
    np_image = image.unsqueeze_(0)
    model.eval()
    
    with torch.no_grad():
        output = model.forward(np_image)
        ps = torch.exp(output)
        k_top,top_classes_idx = ps.topk(topk,dim=1)
        k_top,top_classes_idx = np.array(k_top.to('cpu')[0]),np.array(top_classes_idx.to('cpu')[0])
    
        idx_to_class = {x:y for y,x in model.class_to_idx.items()}
        top_classes = []
        for index in top_classes_idx:
            top_classes.append(idx_to_class[index])
        
        
        return list(k_top), list(top_classes)
    if argp.category_names != None:
        with open(argp.category_names,'r')as f:
            cat_to_name = json.load(f)
            top_class_names = [cat_to_name[top] for top in list(top_classes)]
            print(probabilities:{list(k_top)},classes:{list(top_class_names)})
    else:
        print(probabilities:{list(k_top)},classes:{list(top_classes)})
        
        
if _name_ == '_main_':
    main()
    
