import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from torch import nn
from torchvision.datasets import ImageFolder
import torchvision 
from torch.nn import CrossEntropyLoss
import sys
from torchsummary import summary
import pandas as pd
import argparse
import wandb
from earlystoppingpytorch.pytorchtools import EarlyStopping
import pandas as pd
import numpy as np

# for reading and displaying images
from skimage.io import imread
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# for evaluating the model
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
from sklearn.model_selection import train_test_split



def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    from math import floor
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor( ((h_w[0] + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
    w = floor( ((h_w[1] + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
    return h, w

class Net(nn.Module):   
    def __init__(self,num_layers=1,num_filters=32,filter_step="flat",bn=True,kernal_size=3,stride=1,padding=1,num_linear_layer=2,linear_layer_step=500,dropout=True,dropout_frac=0.25):
        super(Net, self).__init__()
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.cnn_layers = []
        self.dummy_param = nn.Parameter(torch.empty(0))
        
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.filter_step = filter_step
        self.dropout = dropout
        self.bn = bn
        self.kernal_size = kernal_size
        self.stride = stride
        self.padding = padding
        self.num_linear_layer = num_linear_layer
        self.linear_layer_step = linear_layer_step
        self.dropout = dropout
        self.dropout_frac = dropout_frac

        self.setup_cnn_layers()
        self.cnn_layers = Sequential(*self.cnn_layers)
        self.dropout = nn.Dropout(0.25)
        
        
    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        if self.dropout:
            x = self.dropout(x)
        x = self.linear_layers(x)
        return x
    
    def setup_cnn_layers(self):
                
        factor = 1
        
        if self.filter_step == 'inc':
            factor = 2
        elif self.filter_step == 'dec':
            factor = 1/2
        print("factor",factor)
        h,w = (224,224)
        last_layer_filters = self.num_filters
        last_out_channels = 3
        for ix,num in enumerate(range(self.num_layers)):
            self.cnn_layers.append(Conv2d(last_out_channels, last_layer_filters, kernel_size=self.kernal_size, stride=self.stride, padding=(self.padding,self.padding)))
            print("b",h,w)
            h,w = conv_output_shape((h,w),kernel_size=self.kernal_size,stride=self.stride,pad=self.padding)
            print("a",h,w)
            self.cnn_layers.append(ReLU(inplace=True))
            if self.padding < self.kernal_size //2 :
                padding = self.padding
            else:
                padding = 0
            self.cnn_layers.append(MaxPool2d(kernel_size=self.kernal_size, stride=self.stride,padding =padding))
            if self.bn:
                self.cnn_layers.append(BatchNorm2d(last_layer_filters))          
            h,w = conv_output_shape((h,w),kernel_size=self.kernal_size,stride=self.stride,pad=padding)
            print("last h , w" ,last_layer_filters, h , w )
            last_out_channels = last_layer_filters
            last_layer_filters = int(last_layer_filters * factor)
        if self.num_linear_layer == 1:
            self.linear_layers = Sequential(
                Linear(last_out_channels * h * w, 10)
            )
        else:
            self.linear_layers = Sequential(
                Linear(last_out_channels * h * w, self.linear_layer_step),
                Linear( self.linear_layer_step , 10)
            ) 

train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

train_dataset = ImageFolder("inaturalist_12K/train",transform=train_transforms)
test_dataset = ImageFolder("inaturalist_12K/val")

targets = train_dataset.targets


train_idx, valid_idx= train_test_split(
np.arange(len(targets)),
test_size=0.1,
shuffle=True,
stratify=targets)
batch_size = 4
lr = 0.0000005


train_ix  = pd.read_csv('train_ix.csv')['0']
test_ix = pd.read_csv('test_ix.csv')['0']
print(len(train_ix))
train_sampler = torch.utils.data.SubsetRandomSampler(train_ix.tolist())
valid_sampler = torch.utils.data.SubsetRandomSampler(test_ix.tolist())

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,num_workers=2)
valid_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler,num_workers=2)



def train(model,epoch,early_stopping,step=1):
    model.train()
    count=0
    train_loss = 0
    for step, (x_train,y_train) in enumerate(iter(train_loader),start=step):
        model.train()
        x_train = x_train.cuda()
        y_train = y_train.cuda()
        optimizer.zero_grad()
        # prediction for training and validation set
        output_train = model(x_train)

        # computing the training and validation loss
        loss_train = criterion(output_train, y_train)
#         train_losses.append(loss_train)

        loss_train.backward()
        optimizer.step()
        
        tr_loss = loss_train.item()
        train_loss += tr_loss
        tr_loss = tr_loss/ len(y_train)
        
        n_dev_correct = 0
        dev_loss = 0
        val_samples =0 
    with torch.no_grad():
        model.eval()
        for x_valid, y_valid in iter(valid_loader):
            x_valid = x_valid.cuda()
            y_valid = y_valid.cuda()
            answer = model(x_valid)
            n_dev_correct += (torch.max(answer, 1)[1].view(len(y_valid)) == y_valid).sum().item()
            val_samples += len(y_valid)
            dev_loss += criterion(answer, y_valid).item()
        
        
    dev_acc = 100. * n_dev_correct / len(valid_idx)
    dev_loss = dev_loss / len(valid_idx)
    print({"epoch": epoch,
             "training_loss": train_loss/len(train_ix),
             "dev_loss": dev_loss,
             "dev_accuracy": dev_acc,
            })
    wandb.log(
            {"epoch": epoch,
             "training_loss": train_loss/len(train_ix),
             "dev_loss": dev_loss,
             "dev_accuracy": dev_acc,
            }
        )
    early_stopping(dev_loss, model)
        
    if early_stopping.early_stop:
        return step,False
    return step , True 
    #         wanb.log
    
if __name__ == '__main__':
    # main(sys.argv[1:])
    parser = argparse.ArgumentParser(
        usage="%(prog)s [OPTION] [FILE]...",
        description="Print or check SHA1 (160-bit) checksums."
    )
    parser.add_argument('--learning_rate', )
    parser.add_argument("--num_layers")
    parser.add_argument("--num_filters")
    parser.add_argument("--filter_step")
    parser.add_argument("--bn")
    parser.add_argument("--kernal_size")
    parser.add_argument("--stride")
    parser.add_argument("--padding")
    parser.add_argument("--num_linear_layer")
    parser.add_argument("--linear_layer_step")
    parser.add_argument("--dropout")
    parser.add_argument("--dropout_frac")

    args = parser.parse_args()
    wandb.init(project='assignment-2a', entity='raghavan')

    learning_rate = float(args.learning_rate)
    num_layers = int(args.num_layers)
    num_filters = int(args.num_filters)
    filter_step = args.filter_step
    bn = args.bn == "True"
    kernal_size = int(args.kernal_size)
    stride = int(args.stride)
    padding = int(args.padding)
    num_linear_layer = int(args.num_linear_layer)
    linear_layer_step = int(args.linear_layer_step)
    dropout = args.dropout == "True"
    dropout_frac = float(args.dropout_frac)

    wandb.config.batch_size = 4
    wandb.config.learning_rate = learning_rate
    wandb.config.num_layers = num_layers
    wandb.config.num_filters = num_filters
    wandb.config.filter_step = filter_step
    wandb.config.bn = bn
    wandb.config.kernal_size = kernal_size
    wandb.config.stride = stride
    wandb.config.padding = padding
    wandb.config.num_linear_layer = num_linear_layer
    wandb.config.linear_layer_step = linear_layer_step
    wandb.config.dropout = dropout
    wandb.config.dropout_frac = dropout_frac
    model = Net(num_layers=num_layers,num_filters=num_filters,filter_step=filter_step,
        bn=bn,kernal_size=kernal_size,stride=stride,padding=padding,num_linear_layer=num_linear_layer,
        linear_layer_step=linear_layer_step,dropout=dropout,dropout_frac=dropout_frac)

    optimizer = Adam(model.parameters(), lr=lr)
    criterion = CrossEntropyLoss()
#     if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()



    summary(model,(3,224,224))

    n_epochs = 50
    train_losses = []
    val_losses = []
    step = 1
    early_stopping = EarlyStopping(patience=10, verbose=True,delta=0)
    should_continue=True
    for epoch in range(n_epochs):
        if should_continue:
            print(epoch)
            step,should_continue = train(model,epoch,early_stopping,step)
        else:
            print('stopping')
