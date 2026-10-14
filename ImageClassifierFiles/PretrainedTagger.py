import hashlib
import logging
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import clip
import torch
from PIL import Image
from tqdm import tqdm

from .ImageFiles import list_images

log = logging.getLogger("imagesorter.clip")

CLIP_CACHE_DIR = os.path.expanduser("~/.cache/clip")


def is_model_cached(model_name: str = "ViT-B/32") -> bool:
    """True if the CLIP weights for model_name are already downloaded."""
    return os.path.isfile(os.path.join(CLIP_CACHE_DIR, os.path.basename(clip._MODELS[model_name])))


def download_model(model_name: str = "ViT-B/32", progress_callback=None) -> str:
    """
    Download the CLIP weights into the cache (skipped if already present).

    progress_callback(downloaded_bytes, total_bytes) is called as data arrives.
    The file is verified against the SHA256 embedded in OpenAI's download URL.
    Returns the path of the cached file.
    """
    url = clip._MODELS[model_name]
    expected_sha = url.split("/")[-2]
    target = os.path.join(CLIP_CACHE_DIR, os.path.basename(url))
    if os.path.isfile(target):
        return target

    os.makedirs(CLIP_CACHE_DIR, exist_ok=True)
    partial = target + ".part"
    log.info("Downloading CLIP weights %s from %s", model_name, url)
    sha = hashlib.sha256()
    with urllib.request.urlopen(url) as response, open(partial, "wb") as out:
        total = int(response.info().get("Content-Length", 0))
        done = 0
        while True:
            chunk = response.read(1024 * 256)
            if not chunk:
                break
            out.write(chunk)
            sha.update(chunk)
            done += len(chunk)
            if progress_callback:
                progress_callback(done, total)

    if sha.hexdigest() != expected_sha:
        os.remove(partial)
        raise RuntimeError("Downloaded CLIP file failed its checksum; please try again.")
    os.replace(partial, target)
    log.info("CLIP weights saved to %s", target)
    return target


class PretrainedTagger:
    """
    Uses OpenAI CLIP to embed images and text into a shared vector space.

    This enables two capabilities without any custom training:
      - Semantic search: "show me images of sunsets" finds visually matching images.
      - Zero-shot classification: rank candidate labels against an image by similarity.

    Embeddings are normalized L2 vectors, so similarity is a simple dot product.
    """

    def __init__(self, model_name: str = "ViT-B/32", download_progress=None):
        """download_progress(downloaded_bytes, total_bytes) reports the first-run download."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        download_model(model_name, download_progress)
        log.info("Loading CLIP %s on %s", model_name, self.device)
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        log.info("CLIP ready")

    def _load_tensor(self, image_path: str):
        """Decode and preprocess one image; returns None if it can't be read."""
        try:
            with Image.open(image_path) as img:
                return self.preprocess(img.convert("RGB"))
        except Exception as e:
            log.warning("Skipping %s: %s", image_path, e)
            return None

    def _preprocess_image(self, image_path: str) -> torch.Tensor:
        return self.preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(self.device)

    def embed_images(
        self, image_paths: list, progress_callback=None, batch_size: int = 32, workers: int = 4
    ) -> dict:
        """
        Compute normalized CLIP embeddings for a list of image paths.
        Returns {path: embedding_tensor} with tensors stored on CPU.

        Images are decoded on several threads and embedded in batches.
        progress_callback(done, total) is called after each batch.
        """
        embeddings = {}
        total = len(image_paths)
        log.info("Embedding %d images", total)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for start in tqdm(range(0, total, batch_size), desc="Indexing images with CLIP"):
                batch_paths = image_paths[start : start + batch_size]
                loaded = [
                    (p, t) for p, t in zip(batch_paths, pool.map(self._load_tensor, batch_paths))
                    if t is not None
                ]
                if loaded:
                    batch = torch.stack([t for _, t in loaded]).to(self.device)
                    with torch.no_grad():
                        emb = self.model.encode_image(batch)
                        emb = emb / emb.norm(dim=-1, keepdim=True)
                    for (path, _), e in zip(loaded, emb.cpu()):
                        embeddings[path] = e.unsqueeze(0)
                if progress_callback:
                    progress_callback(min(start + batch_size, total), total)
        log.info("Embedded %d of %d images", len(embeddings), total)
        return embeddings

    def index_folder(self, folder: str, progress_callback=None) -> dict:
        """Embed all images in a folder. Returns the embeddings dict."""
        log.info("Indexing folder %s", folder)
        return self.embed_images(list_images(folder), progress_callback)

    def search(self, query: str, embeddings: dict, top_k: int = 20) -> list:
        """
        Find the top_k images most similar to a natural-language query.
        Returns a sorted list of (image_path, similarity_score) tuples.
        """
        text_tokens = clip.tokenize([query]).to(self.device)
        with torch.no_grad():
            text_emb = self.model.encode_text(text_tokens)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
            text_emb = text_emb.cpu()

        scores = {path: (text_emb @ emb.T).item() for path, emb in embeddings.items()}
        log.debug("Search %r over %d images", query, len(embeddings))
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def classify(self, image_path: str, candidate_labels: list) -> tuple:
        """
        Zero-shot classify an image against a list of candidate labels.
        Prompts are formatted as "a photo of <label>" which CLIP handles well.
        Returns (best_label, probabilities_list).
        """
        img_tensor = self._preprocess_image(image_path)
        prompts = [f"a photo of {label}" for label in candidate_labels]
        text_tokens = clip.tokenize(prompts).to(self.device)

        with torch.no_grad():
            logits_per_image, _ = self.model(img_tensor, text_tokens)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy().flatten().tolist()

        best_idx = probs.index(max(probs))
        return candidate_labels[best_idx], probs
