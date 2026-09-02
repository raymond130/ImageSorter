import os
import shutil
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image
from glob import glob

IMAGE_EXTENSIONS = ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG")

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


def copy_images_to_folders(confirmed_predictions: dict, dest_folder: str) -> None:
    """
    Standalone helper: copies images to dest_folder/<label>/ based on a
    {image_path: label} dict. Does not move or delete originals.
    """
    for image_path, label in tqdm(confirmed_predictions.items(), desc="Copying images"):
        dest_dir = os.path.join(dest_folder, label)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy2(image_path, dest_dir)


class ImageClassifier:
    def __init__(
        self,
        loadfolder: str,
        classes: list,
        *,
        model: nn.Module = None,
        imagetransform=None,
    ):
        self.loadfolder = loadfolder
        self.classes = classes
        self.model = model
        self.imgTransform = imagetransform or DEFAULT_TRANSFORM
        self.images: dict = {}
        self.predictions: dict = {}

    def loadImages(self):
        self.images = {}
        for ext in IMAGE_EXTENSIONS:
            pattern = os.path.join(self.loadfolder, ext)
            for image_path in tqdm(glob(pattern), desc=f"Loading {ext}"):
                try:
                    img = Image.open(image_path).convert("RGB")
                    self.images[image_path] = self.imgTransform(img).unsqueeze(0)
                except Exception as e:
                    print(f"Skipping {image_path}: {e}")

    def get_predictions(self):
        if self.model is None:
            raise ValueError("No model set — assign one before calling get_predictions().")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.model.to(device)
        self.predictions = {}
        for image_path, tensor in tqdm(self.images.items(), desc="Predicting"):
            with torch.no_grad():
                outputs = self.model(tensor.to(device))
                probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy().flatten().tolist()
            self.predictions[image_path] = self.classes[probs.index(max(probs))]

    def stage_copies(self, dest_folder: str, confirmed_predictions: dict = None) -> None:
        """Copy images to dest_folder/<label>/ without removing originals."""
        preds = confirmed_predictions if confirmed_predictions is not None else self.predictions
        copy_images_to_folders(preds, dest_folder)

    def move_images(self, dest_folder: str, confirmed_predictions: dict = None) -> None:
        """Move images to dest_folder/<label>/ (destructive — originals removed)."""
        preds = confirmed_predictions if confirmed_predictions is not None else self.predictions
        for image_path, label in tqdm(preds.items(), desc="Moving images"):
            dest_dir = os.path.join(dest_folder, label)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.move(image_path, dest_dir)
