
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm.notebook import tqdm
import ImageDataSet
import ImageClassifierModel

class ImageClassifierTrainer:


    ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Public Variables
    ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ## Folder names

    train_folder = "F:/Sorted pictures test/Train/"
    valid_folder = "F:/Sorted pictures test/Test/"
    test_folder = "F:/Sorted pictures test/Validate/"

    classes : list[str]
    numclasses: int

    train_dataset: ImageDataSet
    val_dataset: ImageDataSet
    test_dataset: ImageDataSet

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader

    model: ImageClassifierModel

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
        self.train_dataset = ImageDataSet(self.train_folder, transform=self.transform)
        self.val_dataset = ImageDataSet(self.valid_folder, self.transform)
        self.test_dataset = ImageDataSet(self.test_folder, self.transform)

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

        model = ImageClassifierModel(num_classes=self.numclasses)
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