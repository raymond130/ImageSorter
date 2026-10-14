import logging
import os
import random
import shutil

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
log = logging.getLogger("imagesorter.splitter")

SPLIT_NAMES = ("train", "val", "test")


def split_dataset(
    source_folder: str,
    dest_folder: str,
    ratios: tuple = (0.7, 0.15, 0.15),
    seed: int = 42,
    move: bool = False,
) -> dict:
    """
    Randomly split a folder of sorted images into train / val / test sets.

    source_folder must contain one subfolder per category (the layout
    ImageFolder expects). The result is written to:
        dest_folder/train/<category>/...
        dest_folder/val/<category>/...
        dest_folder/test/<category>/...

    The split is done per category, so every set sees roughly the same class
    balance. Images are copied by default; pass move=True to relocate them.
    Returns {split_name: image_count}.
    """
    if len(ratios) != 3 or any(r < 0 for r in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("ratios must be three non-negative numbers that sum to 1.")
    if not os.path.isdir(source_folder):
        raise FileNotFoundError(f"Source folder not found: {source_folder}")
    if os.path.abspath(dest_folder).startswith(os.path.abspath(source_folder) + os.sep):
        raise ValueError("Destination must not be inside the source folder.")
    for name in SPLIT_NAMES:
        existing = os.path.join(dest_folder, name)
        if os.path.isdir(existing) and os.listdir(existing):
            raise FileExistsError(f"{existing} already contains files; choose an empty destination.")

    categories = sorted(
        d for d in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, d))
    )
    if not categories:
        raise ValueError("Source folder has no category subfolders.")

    rng = random.Random(seed)
    transfer = shutil.move if move else shutil.copy2
    counts = {name: 0 for name in SPLIT_NAMES}

    for category in categories:
        cat_dir = os.path.join(source_folder, category)
        images = sorted(
            f for f in os.listdir(cat_dir) if f.lower().endswith(IMAGE_EXTENSIONS)
        )
        rng.shuffle(images)

        n = len(images)
        n_val = round(n * ratios[1])
        n_test = round(n * ratios[2])
        # Training always keeps the remainder, so it is never empty for a non-empty category.
        n_train = n - n_val - n_test
        groups = {
            "train": images[:n_train],
            "val": images[n_train : n_train + n_val],
            "test": images[n_train + n_val :],
        }

        for split, files in groups.items():
            # Create the category folder even if empty so all three sets share the same classes.
            out_dir = os.path.join(dest_folder, split, category)
            os.makedirs(out_dir, exist_ok=True)
            for f in files:
                transfer(os.path.join(cat_dir, f), out_dir)
            counts[split] += len(files)

    log.info("Split %s -> %s (ratios %s, seed %s, move=%s): %s", source_folder, dest_folder, ratios, seed, move, counts)
    return counts
