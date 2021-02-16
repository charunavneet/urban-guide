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
%matplotlib inline

parser = argparse.ArgumentParser()
parser.add_argument('--arch',dest="arch",action='store',default='vgg19',type=str)
parser.add_argument('--data_dir',type=str,default='flowers/',help='directory of the files")
parser.add_argument('--save_dir',dest='save_dir',action="store",default='./checkpoint.pth',help='directory to save model')
parser.add_argument('--learning_rate',dest='learning_rate',action='store',default=0.001,help='learning rate of the model')
parser.add_argument('--hidden_units',dest='hidden_units',type=int,action='store',default=[4096,1024],help='hidden units of the model')
parser.add_argument('--epochs',dest='epochs',action='store',type=int,default=15,help='epochs to train the model')
parser.add_argument('--gpu',dest='gpu',action='store',default=False)
argp=parser.parse_args()
def dataloader(data_dir):
    data_transforms = {
    'train': transforms.Compose([transforms.RandomResizedCrop(size=224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                                ]),
    'valid':transforms.Compose([transforms.RandomResizedCrop(size=256),
                                transforms.CenterCrop(size=224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                                ]),
    'test':transforms.Compose([transforms.RandomResizedCrop(size=256),
                                transforms.CenterCrop(size=224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                                ]),
    }                    
    image_datasets = { 'train': datasets.ImageFolder(train_dir,transform = data_transforms['train']),
                 'valid': datasets.ImageFolder(valid_dir,transform = data_transforms['valid']),
                 'test': datasets.ImageFolder(test_dir,transform = data_transforms['test'])
                 } 
    data_loader = {'train':torch.utils.data.DataLoader(image_datasets['train'],batch_size = 64,shuffle=True),
              'valid':torch.utils.data.DataLoader(image_datasets['valid'],batch_size = 64,shuffle=True),
              'test':torch.utils.data.DataLoader(image_datasets['test'],batch_size = 64,shuffle=True)
              }
                    
    return data_loader
                   
def model(hidden_units,learning_rate,arch,device="cpu"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.vgg19(pretrained = True)

    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(OrderedDict([
        ('fc1',nn.Linear(25088,4096)),
        ('relu1',nn.ReLU()),
        ('dropout1',nn.Dropout(p=0.5)),
        ('fc2',nn.Linear(4096,1024)),
        ('relu2',nn.ReLU()),
        ('dropout2',nn.Dropout(p=0.5)),
        ('fc3',nn.Linear(1024,102)),
        ('output',nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(),lr=0.001)
    model.to(device)
    return (model,criterion,optiumizer)
                    
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loader = data_loader(argp.data_dir)
    model,criterion,optimizer = model(hidden_units=argp.hidden_units,learning_rate=argp.learning_rate,arch=argp.arch,device)
                    
    epochs = 15


    for e in range (epochs):
        print("Epoch: {}/{}".format(e+1,epochs))
        model.train()
        running_loss=0.0
    
    
        for i, (inputs,labels) in enumerate(data_loader['train']):
            inputs,labels = inputs.to(device),labels.to(device)
            optimizer.zero_grad()
            output=model.forward(inputs)
            loss=criterion(output,labels)
            loss.backward()
            optimizer.step()
        
            running_loss +=loss.item() * inputs.size(0)
            print("Train Loss: {:.4f}".format(loss.item()))
        
    valid_loss = 0.0
    valid_accuracy =0.0
    with torch.no_grad():
        model.eval()
        
        for ii,(inputs,labels) in enumerate(data_loader['valid']):
            optimizer.zero_grad()
            inputs,labels = inputs.to(device),labels.to(device)
            output=model.forward(inputs)
            loss=criterion(output,labels)
            valid_loss += loss.item() * inputs.size(0)
            
            ret,predictions = torch.max(output.data,1)
            equality = predictions.eq(labels.data.view_as(predictions))
            acc = torch.mean(equality.type(torch.FloatTensor))
            valid_accuracy += acc.item() * inputs.size(0)
            print("validation loss:{:.4f}".format(loss.item()),
                    "Accuracy: {:.4f}".format(acc.item()))
                    
                    
    model.class_to_idx = image_datasets['train'].class_to_idx
    checkpoint = {'structure':'vgg19',
                 'class_to_idx':model.class_to_idx,
                 'layers':[25088,4096,1024,102],
                 'dropout': '0.5',
                 'epochs': 15,
                 'state_dict':model.state_dict,
                'optimizer_state_dict': optimizer.state_dict()
                }

    torch.save(checkpoint,'model_checkpoint.pth')
                    
if _name_ =='_main_':
    main()                    
            
                            
                                        
                                        