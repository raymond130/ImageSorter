## How ImageSorter works

ImageSorter helps you find and organize photos on your own computer. Everything runs locally:
your photos are never uploaded anywhere. You type folder paths (for example
`C:/Users/you/Pictures`), and the app reads from them directly.

It has three tools, one per tab.

---

### 1. Search

Find photos by describing them, with no training needed.

1. Enter a **folder** of images.
2. The first time, the app downloads the CLIP model (about 350 MB, one time only) and then
   **indexes** your folder. A progress bar shows both steps. Indexing is done once per folder per session.
3. Type a description, such as *"sunset over water"* or *"a dog on the beach"*, and the best
   matches appear, with a similarity score.

Only JPG and PNG files directly inside the folder are searched (subfolders are not included).

### 2. Train Custom Model

Teach the app **your own categories** (for example *Chester*, *Memes*, *Nature*) by example.

1. Sort a sample of your photos into one subfolder per category:
   ```
   sorted_photos/
       Chester/   Nature/   Memes/
   ```
2. Open **"Split a sorted folder"** to randomly divide that sample into **train / validation / test**
   folders. Your originals are copied, not moved. The folder boxes below are filled in for you.
   - *Train*: what the model learns from.
   - *Validation*: checked after every epoch to see whether it is learning in general.
   - *Test*: held back for final checking.
3. Choose the number of **epochs** (passes over the training data) and the **learning rate**.
   The defaults (5 epochs, 0.001) are a reasonable start. If the validation loss stops improving
   or rises while training loss falls, the model is memorizing; use fewer epochs or more photos.
4. Choose where to save the model (a folder and a file name; the default is the `models` folder next
   to the app). If the file already exists you will see a warning that it will be overwritten.
5. Click **Train Model**. A progress bar, a loss chart and a per-epoch summary update as it runs.

More photos per category and a balanced number across categories generally give better results.
Training is much faster on a computer with an NVIDIA GPU, but works on a CPU.

### 3. Sort Images

Apply a model to a folder of unsorted photos, check the result, and copy or move them into category folders.

1. Enter the **folder of unsorted images** and a **destination folder**.
2. Pick a model:
   - **Freshly trained**: the model you just trained in this session.
   - **Pretrained CLIP (zero-shot)**: no training. Type your own labels separated by commas
     (for example `Chester, Nature, Food`) and CLIP picks the best match for each photo.
   - **Load from file**: a `.pt` model you saved earlier.
3. Click **Run Model on Folder**. A progress bar shows how many photos are done.
4. **Review the results.** Photos are shown as small previews, 24 per page. Change the label on any
   photo that was put in the wrong category. Your changes are kept as you move between pages.
5. Choose **Copy** or **Move**, then click the button. Photos go into `destination/<category>/`.
   - **Copy** (the default) leaves your original photos untouched.
   - **Move** removes them from the source folder, so you must tick a confirmation box first.
   - Either way, a photo is never overwritten: if the destination already has a file with the same
     name, the new one is saved as `name (1).jpg`.

   **Every photo is verified.** Each one is copied, then checked against the original (by file size,
   or by comparing contents if you tick **Deep verify**, which is slower). With **Move**, the original
   is deleted only after its copy has passed the check, so a failed transfer never loses a photo: the
   bad copy is discarded and the original stays where it was. When everything is done, the app checks
   the destination once more and reports how many photos were verified, and lists any that were not.

   A record of every photo (original location, new location, category, size and result) is saved as
   `sort_manifest_<date>_<time>.csv` in the destination folder. You can open it in Excel to trace
   where any photo went.

---

### Tips and troubleshooting

- **Paths**: copy a folder path from File Explorer's address bar. Quotes around a path are fine.
- **Slow first start**: the first Search or CLIP sort downloads the CLIP model. After that it is
  cached on your computer and works offline.
- **Large folders**: indexing and sorting hundreds of photos can take a few minutes on a CPU. The
  progress bar shows how far along it is.
- **"Folder not found"**: check the path and that the folder is not on a disconnected drive.
- **A photo is skipped**: unreadable or corrupt files are skipped and noted in the debug log.
- **Something went wrong?** Open the **Debug log** section at the bottom of this page. It records major
  events (downloads, indexing, training epochs, saves, copies) and any errors, and is also saved to
  `logs/imagesorter.log` next to the app. Include it if you report a problem.
