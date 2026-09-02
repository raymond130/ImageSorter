import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
from .ImageDataSet import ImageDataSet
from .ImageClassifierModel import ImageClassifierModel

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


class ImageClassifierTrainer:
    def __init__(
        self,
        train_folder: str,
        val_folder: str,
        test_folder: str,
        transform=None,
    ):
        self.train_folder = train_folder
        self.val_folder = val_folder
        self.test_folder = test_folder
        self.transform = transform or DEFAULT_TRANSFORM
        self.model: ImageClassifierModel = None

        self.classes = sorted([
            name for name in os.listdir(train_folder)
            if os.path.isdir(os.path.join(train_folder, name))
        ])
        self.numclasses = len(self.classes)

        self.train_dataset = ImageDataSet(train_folder, transform=self.transform)
        self.val_dataset = ImageDataSet(val_folder, transform=self.transform)
        self.test_dataset = ImageDataSet(test_folder, transform=self.transform)

        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=32, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=32, shuffle=False)

    def train(self, num_epochs: int = 5, learning_rate: float = 0.001, progress_callback=None):
        """
        Train the model. Does NOT auto-run on init — call explicitly.

        progress_callback(epoch, total_epochs, train_loss, val_loss) is called
        after each epoch if provided (useful for Streamlit progress bars).
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = ImageClassifierModel(num_classes=self.numclasses)
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        train_losses, val_losses = [], []

        for epoch in range(num_epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            for images, labels in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs} — Train"):
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
                for images, labels in tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{num_epochs} — Val"):
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    running_loss += loss.item() * labels.size(0)
            val_loss = running_loss / len(self.val_loader.dataset)
            val_losses.append(val_loss)

            print(f"Epoch {epoch+1}/{num_epochs} — Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")
            if progress_callback:
                progress_callback(epoch + 1, num_epochs, train_loss, val_loss)

        self.model = model
        return train_losses, val_losses

    def save(self, path: str) -> None:
        """Save model weights and class list to a .pt checkpoint file."""
        if self.model is None:
            raise RuntimeError("No model to save — call train() first.")
        torch.save({"state_dict": self.model.state_dict(), "classes": self.classes}, path)

    @staticmethod
    def load_model(path: str) -> tuple:
        """
        Load a saved model checkpoint. Returns (model, classes).
        The model is returned in eval mode, ready for inference.
        """
        checkpoint = torch.load(path, map_location="cpu")
        classes = checkpoint["classes"]
        model = ImageClassifierModel(num_classes=len(classes))
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model, classes
