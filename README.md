# ImageSorter

A local app for searching and sorting your photos.

- **Search** – type a description ("sunset over water") and find matching images in a folder, using OpenAI CLIP.
- **Train Custom Model** – sort a sample of your photos into category folders and train a small classifier on them. A built-in helper can split a sorted folder into train / validation / test sets for you.
- **Sort Images** – run a trained model (or zero-shot CLIP with labels you type) over a folder, review and correct the predictions, then copy (or, after confirming, move) the images into per-category folders. Copying is the default and leaves originals untouched. Every file is verified after transfer (size, or contents with "Deep verify"), a move deletes an original only after its copy is confirmed, and a CSV manifest of what went where is saved in the destination folder.

See [HELP.md](HELP.md) for a walkthrough of each tab (it is also shown in the app's **Help** tab).

## Install

Requires Python 3.9+ and git.

```
pip install -r requirements.txt
```

> If you previously installed the PyPI package named `clip` (a clipboard manager), run
> `pip uninstall clip` first. It shares a module name with OpenAI CLIP and will break the Search tab.

## Run

```
streamlit run app.py
```

ImageSorter is designed to run **on your own machine**. Folders are entered as paths and read/written
directly on your disk. Hosting it on a remote server would only give it access to the server's files.
The first run downloads the CLIP weights (~350 MB), and the first training run downloads
the pretrained EfficientNet weights (~20 MB). After that it works offline.

## Training data layout

```
sorted_photos/
    Chester/  a.jpg b.jpg ...
    Nature/   ...
    Food/     ...
```

Use the "Split a sorted folder" section in the Train tab to produce `train/`, `val/` and `test/`
folders with this same category structure.

## Notes

- Logs: major events and errors are written to `logs/imagesorter.log` (also viewable in the Help tab).
- Trained models are saved to the `models/` folder by default; you can choose any folder in the Train tab.
- Supported formats: JPG and PNG.
- Saved `.pt` model files are loaded with `weights_only=True`, but still only load models you trust.
