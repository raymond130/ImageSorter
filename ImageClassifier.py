
#App is meant to sort a users images based on folders they define
#includes a dataset loader, model set up and transformation for the rest of the images

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.models
from torchvision.models import resnet50, ResNet50_Weights
import timm

import matplotlib.pyplot as plt # For data viz
import pandas as pd
import numpy as np
import sys
from tqdm.notebook import tqdm


from PIL import Image
from glob import glob

import shutil

#preliminary classes
class ImageDataset(Dataset):

    def __init__(self, data_dir, transform = None):
        self.data = ImageFolder(data_dir, transform=transform) 


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    @property
    def classes(self):
        return self.data.classes

        #image = transforms.Compose([
         #   transforms.Resize((128, 128)),
          #  transforms.ToTensor(),])(Image.open(image_path))

class SimpleImageClassifierModel(nn.Module):
    def __init__(self, num_classes=0):
        super(SimpleImageClassifierModel, self).__init__()
        # Where we define all the parts of the model
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        enet_out_size = 1280
        # Make a classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(enet_out_size, num_classes)
        )
    
    def forward(self, x):
        # Connect these parts and return the output
        x = self.features(x)
        output = self.classifier(x)
        return output

class ClassifierTrainer:


    ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Public Variables
    ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ## Folder names

    train_folder = "F:/Sorted pictures test/Train/"
    valid_folder = "F:/Sorted pictures test/Test/"
    test_folder = "F:/Sorted pictures test/Validate/"

    classes : list[str]
    numclasses: int

    train_dataset: ImageDataset
    val_dataset: ImageDataset
    test_dataset: ImageDataset

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader

    model: SimpleImageClassifierModel

    transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),])

    def __init__(self):
        ## need class names and number of classes, from the OS
        
        self.classes = []
        self.numclasses = 0

        for name in os.listdir(self.train_folder):
            self.classes.append(name)
            self.numclasses += 1

        ## Datasets
        self.train_dataset = ImageDataset(self.train_folder, transform=self.transform)
        self.val_dataset = ImageDataset(self.valid_folder, self.transform)
        self.test_dataset = ImageDataset(self.test_folder, self.transform)

        ## Loaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=32, shuffle=False)
        self.test_loader = DataLoader(
            self.val_dataset, batch_size=32, shuffle=False)

        self.train_model()

    def train_model(self, num_epochs = 5):

        # Simple training loop
        train_losses, val_losses = [], []

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = SimpleImageClassifierModel(num_classes=self.numclasses)
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(num_epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            for images, labels in tqdm(self.train_loader, desc='Training loop'):
                # Move inputs and labels to the device
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * labels.size(0)
            train_loss = running_loss / len(self.train_loader.dataset)
            train_losses.append(train_loss)


            # Validation phase
            model.eval()
            running_loss = 0.0
            with torch.no_grad():
                for images, labels in tqdm(self.val_loader, desc='Validation loop'):
                    # Move inputs and labels to the device
                    images, labels = images.to(device), labels.to(device)
                
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    running_loss += loss.item() * labels.size(0)
            val_loss = running_loss / len(self.val_loader.dataset)
            val_losses.append(val_loss)
            print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss}, Validation loss: {val_loss}")
        self.model = model


 
    ### Now we have a model that we can use to get predictions from

    ## output of the model is three components: Loading in image from the desired folder, sending to machine then passing through the model and translating to class, then moving to the appropriate folder


    ## Loading in images from unlabelled folder

class ImageSorter:



        loadfolder: str
        destinationFolders : str
        classes: list[str]
        images: dict[str,torch.Tensor]
        predictions : dict[str,str] = {}
        model: nn.Module
        imgTransform = transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
            ])

        def __init__(self,loadfolder, classes: list[str], imagetransform, model: nn.Module = None):
            #initialize the model, folders and classes
            self.loadfolder = loadfolder
            self.classes = classes
            self.model = model
            self.images = {}
            self.predictions = {}
            self.imgTransform = imagetransform

            
            self.destinationFolders = "F:\\Sorted pictures test\\Sorted Test\\"

        def loadImages(self):

            try:
                carl = glob("F:\\Sorted pictures test\\Unsorted\\" + "*.JPG")
            except:
                print("exception!")
            for image in tqdm(glob(self.loadfolder + "*.JPG")):
                temp = Image.open(image).convert('RGB')
                self.images[image] = self.imgTransform(temp).unsqueeze(0)
            for image in tqdm(glob(self.loadfolder + "*.PNG")):
                temp = Image.open(image).convert('RGB')
                self.images[image] = self.imgTransform(temp).unsqueeze(0)

        def get_predictions(self):
            ## send image through model
            self.model.eval()
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            for image in tqdm(self.images.keys()):
                image_tensor = self.images[image]
                with torch.no_grad():
                    image_tensor = image_tensor.to(device)
                    outputs = self.model(image_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy().flatten().tolist()
                    #probabilities = torch.max(probabilities,1)

                    self.predictions[image] = self.classes[list.index(probabilities, max(probabilities))]
                    debug = 0
            
        def move_images(self):
            for image in tqdm(self.images.keys(),desc="Moving images"):
                if not os.path.exists(os.path.join(self.destinationFolders,self.predictions[image])): os.mkdir(os.path.join(self.destinationFolders,self.predictions[image]))
                shutil.move(image,os.path.join(self.destinationFolders,self.predictions[image]) )



trainer = ClassifierTrainer()
model = trainer.model


imagesFolder = "F:\\Sorted pictures test\\Unsorted\\"
#classes = ["Chester","Memes","Misc","Nature","People"]




bongus = resnet50(weights=ResNet50_Weights.DEFAULT)

#transform = ResNet50_Weights.DEFAULT.transforms() 

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),])

classes =ImageDataset("F:\Sorted pictures test\Test",transform=transform).classes

bingus = ImageSorter(imagesFolder, classes, model, imagetransform= transform)          
bingus.loadImages()
bingus.get_predictions()
bingus.move_images()

