import os

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def list_images(folder: str) -> list:
    """
    Return the paths of all JPG/PNG images directly inside folder.

    Matches extensions case-insensitively in a single pass. (Globbing "*.png"
    and "*.PNG" separately returns every file twice on Windows, whose
    filesystem is case-insensitive.)
    """
    return sorted(
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if name.lower().endswith(IMAGE_EXTENSIONS)
        and os.path.isfile(os.path.join(folder, name))
    )
