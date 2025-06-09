#App is meant to sort a users images based on folders they define
#includes a dataset loader, model set up and transformation for the rest of the images

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm.notebook import tqdm
from PIL import Image
from glob import glob

import shutil

class ImageClassifier:
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

