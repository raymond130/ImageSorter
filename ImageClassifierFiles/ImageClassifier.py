import logging
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from .FileTransfer import copy_images_to_folders, move_images_to_folders, sort_images_to_folders  # noqa: F401 (re-exported)
from .ImageFiles import list_images

log = logging.getLogger("imagesorter.classifier")

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])



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
        self.predictions: dict = {}

    def _load_tensor(self, image_path: str):
        """Decode and transform one image; returns None if it can't be read."""
        try:
            with Image.open(image_path) as img:
                return self.imgTransform(img.convert("RGB"))
        except Exception as e:
            log.warning("Skipping %s: %s", image_path, e)
            return None

    def get_predictions(self, progress_callback=None, batch_size: int = 32, workers: int = 4):
        """
        Predict a label for every image in loadfolder.

        Images are decoded on demand in batches (several threads at a time), so
        the whole folder is never held in memory. progress_callback(done, total)
        is called after each batch if provided.
        """
        if self.model is None:
            raise ValueError("No model set — assign one before calling get_predictions().")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.model.to(device)
        self.predictions = {}

        paths = list_images(self.loadfolder)
        total = len(paths)
        log.info("Predicting %d images from %s", total, self.loadfolder)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for start in tqdm(range(0, total, batch_size), desc="Predicting"):
                batch_paths = paths[start : start + batch_size]
                loaded = [
                    (p, t) for p, t in zip(batch_paths, pool.map(self._load_tensor, batch_paths))
                    if t is not None
                ]
                if loaded:
                    batch = torch.stack([t for _, t in loaded]).to(device)
                    with torch.no_grad():
                        indices = self.model(batch).argmax(dim=1).tolist()
                    for (path, _), idx in zip(loaded, indices):
                        self.predictions[path] = self.classes[idx]
                if progress_callback:
                    progress_callback(min(start + batch_size, total), total)
        log.info("Predicted %d of %d images", len(self.predictions), total)

    def stage_copies(self, dest_folder: str, confirmed_predictions: dict = None) -> None:
        """Copy images to dest_folder/<label>/ without removing originals."""
        preds = confirmed_predictions if confirmed_predictions is not None else self.predictions
        copy_images_to_folders(preds, dest_folder)

    def move_images(self, dest_folder: str, confirmed_predictions: dict = None) -> None:
        """Move images to dest_folder/<label>/ (destructive — originals removed)."""
        preds = confirmed_predictions if confirmed_predictions is not None else self.predictions
        move_images_to_folders(preds, dest_folder)
