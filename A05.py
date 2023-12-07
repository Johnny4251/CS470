# Author: John Pertell
# Date:   12.3.23
# Desc:   This program is used in conjunction of Train_A05.py and Eval_A05.py.
#         It provides functions to create two different neural network models
#         designed for the CIFAR10 dataset.

import torch
from torch import nn 
from torchvision.transforms import v2

# Returns the network names
def get_approach_names():
    return ["SuperAugNet", "SimpleAugNet"]

# Returns the description of each network
def get_approach_description(approach_name):
    if approach_name == "SuperAugNet":
        return "A neural network that uses a significant amount of data augmentation for feature extraction.\nThis network has many nodes and uses batch normalization on their inputs."
    elif approach_name == "SimpleAugNet":
        return "A neural network that uses less data augmentation and a lower batch size than SuperAugNet. This network achieves better generalization and does not use batch normalization. This network also has less nodes which helps prevent overfitting."

# Given an approach name, return the corresponding data transform on training
# data. If it is not training(testing) return the testing data transform.
def get_data_transform(approach_name, training):
    if not training:
        data_transform = v2.Compose([
            v2.ToTensor(),
            v2.ConvertImageDtype()
        ])

    # SuperAugNet - is distinct because it uses a lot of data augmentation
    if approach_name == "SuperAugNet":
        data_transform = v2.Compose([
                v2.RandomAdjustSharpness(sharpness_factor=1.5),
                v2.RandomHorizontalFlip(),
                v2.RandomRotation(degrees=15),
                v2.ColorJitter(brightness=0.2, contrast=0.2),
                v2.ToImageTensor(),
                v2.ConvertImageDtype()
            ])
    
    # SimpleAugNet - Uses less data augmentation then SuperAugNet
    elif approach_name == "SimpleAugNet":
        data_transform = v2.Compose([v2. RandomHorizontalFlip(), 
                                     v2.RandomRotation(degrees=15),
                                     v2.ToImageTensor(), 
                                     v2.ConvertImageDtype()])
    
    return data_transform

# Returns the corresponding batch size given what approach
def get_batch_size(approach_name):
    if approach_name == "SuperAugNet":
        return 128
    elif approach_name == "SimpleAugNet":
        return 32

# Creates both models and returns based on approach name
def create_model(approach_name, class_cnt):
    class SuperAugNet(nn.Module):
        def __init__(self, class_cnt):
            super().__init__()        
            self.net_stack = nn.Sequential(

                # Convolutional layers using ReLU for activation
                # and batch normalization.
                nn.Conv2d(3, 64, 3, padding="same"),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64 ,64, 3, padding="same"),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64 ,64, 3, padding="same"),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(3, 3),
                
                nn.Conv2d(64, 128, 3, padding="same"),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, 3, padding="same"),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(3, 3),

                nn.Conv2d(128, 128, 3, padding="same"),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, 3, padding="same"),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),

                nn.Flatten(),

                # Fully connected layers 
                nn.Linear(128, 256),
                nn.BatchNorm1d(256),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                
                nn.Linear(256, class_cnt) # Output layer

            )
        def forward(self, x):        
            logits = self.net_stack(x)
            return logits
    
    class SimpleAugNet(nn.Module):
        def __init__(self, class_cnt):
            super().__init__()      
            self.net_stack = nn.Sequential(

                # Convolutional layers using ReLU for activation
                nn.Conv2d(3, 32, 3, padding="same"),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                nn.Conv2d(32, 32, 3, padding="same"),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Flatten(),
                
                # Fully connected layer
                nn.Linear(2048, 128),
                nn.Linear(128, class_cnt) # Output layer

            )
        def forward(self, x):        
            logits = self.net_stack(x)
            return logits
            
    if approach_name == "SuperAugNet":
        return SuperAugNet(class_cnt)
    elif approach_name == "SimpleAugNet":
        return SimpleAugNet(class_cnt)
    
# Function to train one epoch of model
def train_one_epoch(train_dataloader, model, device, loss_fn, optimizer):
    size = len(train_dataloader.dataset)
    model.train()
    
    # Training on data
    for batch, (X,y) in enumerate(train_dataloader):
        X = X.to(device)
        y = y.to(device)
        
        # Grabbing prediction and loss data
        pred = model(X)
        loss = loss_fn(pred, y)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if batch%100 == 0:
            loss = loss.item()
            index = (batch+1)*len(X)
            print(index, "of", size, ": Loss =", loss)

# Function to test the current state of the model
def test(dataloader, model, device, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for X,y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
    
    print("Training:")
    print("\tAccuracy:", correct)
    print("\tLoss:", test_loss)

# This function is designed to train SuperAugNet or SimpleAugNet
def train_model(approach_name, model, device, train_dataloader, test_dataloader):
    print("\n" + approach_name +":")

    # Using cross entropy as loss function and Adam for optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Both SuperAugNet and SimpleAugNet use different epoch sizes
    if approach_name == "SuperAugNet":
        total_epochs = 40
    elif approach_name == "SimpleAugNet":
        total_epochs = 20
    else:
        total_epochs = 5 # default
        
    # Iterating through each epoch
    for epoch in range(total_epochs):
        print("** EPOCH", (epoch+1), "**")
        train_one_epoch(train_dataloader, model, device, loss_fn, optimizer)
        test(train_dataloader, model, device, loss_fn) # Test train set
        test(test_dataloader, model, device, loss_fn)  # Test test set

    # Returning the pretrained model
    return model
    