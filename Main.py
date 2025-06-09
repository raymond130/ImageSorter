#App is meant to sort a users images based on folders they define
#includes a dataset loader, model set up and transformation for the rest of the images

import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import ImageClassifier
import ImageDataSet
import ImageClassifierTrainer

#Define the model trainer and get the model
###TODO: make classifiertrainer configuration class, and add function to star training (currently default settings are used)
trainer = ImageClassifierTrainer()
model = trainer.model

DefaultTransform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),])


##TODO: Handle this in the gui
imagesFolder = "F:\\Sorted pictures test\\Unsorted\\"
#classes = ["Chester","Memes","Misc","Nature","People"]


##TODO: Move default class instantiation to the image classifier file
bongus = resnet50(weights=ResNet50_Weights.DEFAULT)

##TODO: add default mdoel
#transform = ResNet50_Weights.DEFAULT.transforms() 


##TODO: Move this up or something
classes =ImageDataSet("F:\Sorted pictures test\Test",transform=DefaultTransform).classes

##TODO: These would all be under UI triggers in replit
bingus = ImageClassifier(imagesFolder, classes, model, imagetransform= DefaultTransform)          
bingus.loadImages()
bingus.get_predictions()
bingus.move_images()

