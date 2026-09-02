import clip
import torch
from PIL import Image
from pathlib import Path
from glob import glob
from tqdm import tqdm

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]


class PretrainedTagger:
    """
    Uses OpenAI CLIP to embed images and text into a shared vector space.

    This enables two capabilities without any custom training:
      - Semantic search: "show me images of sunsets" finds visually matching images.
      - Zero-shot classification: rank candidate labels against an image by similarity.

    Embeddings are normalized L2 vectors, so similarity is a simple dot product.
    """

    def __init__(self, model_name: str = "ViT-B/32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

    def _preprocess_image(self, image_path: str) -> torch.Tensor:
        return self.preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(self.device)

    def embed_images(self, image_paths: list) -> dict:
        """
        Compute normalized CLIP embeddings for a list of image paths.
        Returns {path: embedding_tensor} with tensors stored on CPU.
        """
        embeddings = {}
        for path in tqdm(image_paths, desc="Indexing images with CLIP"):
            try:
                img_tensor = self._preprocess_image(path)
                with torch.no_grad():
                    emb = self.model.encode_image(img_tensor)
                    emb = emb / emb.norm(dim=-1, keepdim=True)
                embeddings[path] = emb.cpu()
            except Exception as e:
                print(f"Skipping {path}: {e}")
        return embeddings

    def index_folder(self, folder: str) -> dict:
        """Embed all images in a folder. Returns the embeddings dict."""
        paths = [
            str(p)
            for ext in IMAGE_EXTENSIONS
            for p in Path(folder).glob(f"*{ext}")
        ]
        return self.embed_images(paths)

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
