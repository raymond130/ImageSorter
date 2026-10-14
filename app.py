import logging
import sys
import os
import subprocess
from pathlib import Path

if __name__ == "__main__":
    import streamlit.runtime
    if not streamlit.runtime.exists():
        subprocess.run([sys.executable, "-m", "streamlit", "run", __file__] + sys.argv[1:])
        sys.exit()

import streamlit as st
import torch
import torchvision.transforms as transforms

# Streamlit's file watcher crashes (harmlessly, but noisily) when it inspects
# torch.classes. An empty __path__ makes the watcher skip it.
torch.classes.__path__ = []

from PIL import Image
from ImageClassifierFiles.ImageClassifier import ImageClassifier, sort_images_to_folders
from ImageClassifierFiles.ImageClassifierTrainer import ImageClassifierTrainer
from ImageClassifierFiles.PretrainedTagger import PretrainedTagger, download_model
from ImageClassifierFiles.DatasetSplitter import split_dataset
from ImageClassifierFiles.ImageFiles import list_images
from ImageClassifierFiles.logging_setup import setup_logging, read_log_tail, LOG_FILE

setup_logging()
log = logging.getLogger("imagesorter.app")

THUMB_SIZE = 256
PREVIEW_PAGE_SIZE = 24
APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
HELP_FILE = APP_DIR / "HELP.md"


def get_tagger(slot) -> PretrainedTagger:
    """
    Return the shared CLIP tagger, creating it on first use. While that happens,
    `slot` (an st.empty()) shows a progress bar: first the one-time weights
    download (~350 MB), then loading the model into memory.
    """
    if "tagger" in st.session_state:
        return st.session_state["tagger"]

    bar = slot.progress(0.0, text="Preparing CLIP model...")

    def on_download(done, total):
        if total:
            bar.progress(
                done / total,
                text=f"Downloading CLIP model (first run only)... {done / 1e6:.0f} / {total / 1e6:.0f} MB",
            )

    download_model(progress_callback=on_download)
    bar.progress(1.0, text="Loading CLIP model into memory...")
    tagger = PretrainedTagger()
    st.session_state["tagger"] = tagger
    slot.empty()
    return tagger


# ── Sort preview ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, max_entries=2000)
def load_thumbnail(path: str, size: int = THUMB_SIZE) -> Image.Image:
    """Load an image downscaled to fit in size x size (cached across reruns)."""
    with Image.open(path) as img:
        img.thumbnail((size, size))
        return img.convert("RGB")


def _record_correction(path: str) -> None:
    """Widget callback: remember the user's label for one image."""
    st.session_state["sort_corrections"][path] = st.session_state[f"sortimg_{path}"]


def store_predictions(predictions: dict, classes: list, dest: str) -> None:
    """Save a fresh set of predictions and discard any earlier corrections."""
    for key in [k for k in st.session_state if k.startswith("sortimg_")]:
        del st.session_state[key]
    st.session_state["sort_corrections"] = {}
    st.session_state["sort_predictions"] = predictions
    st.session_state["sort_classes"] = classes
    st.session_state["sort_dest_confirmed"] = dest


def render_sort_preview(predictions: dict, classes: list) -> dict:
    """
    Show the model's predictions as a paged grid of thumbnails and let the
    user correct labels before copying.

    Returns {image_path: confirmed_label} for ALL images, not just the visible
    page. Corrections are stored in st.session_state["sort_corrections"] because
    Streamlit discards widget state for widgets that are not rendered (i.e. on
    other pages).
    """
    if not predictions:
        return {}

    corrections = st.session_state.setdefault("sort_corrections", {})
    paths = list(predictions)
    n_pages = max(1, -(-len(paths) // PREVIEW_PAGE_SIZE))
    page = 1
    if n_pages > 1:
        page = st.number_input("Page", min_value=1, max_value=n_pages, value=1, step=1,
                               help=f"{PREVIEW_PAGE_SIZE} images per page")

    cols_per_row = 4
    page_paths = paths[(page - 1) * PREVIEW_PAGE_SIZE : page * PREVIEW_PAGE_SIZE]
    for row_start in range(0, len(page_paths), cols_per_row):
        cols = st.columns(cols_per_row)
        for col, path in zip(cols, page_paths[row_start : row_start + cols_per_row]):
            with col:
                try:
                    st.image(load_thumbnail(path), use_container_width=True)
                except Exception:
                    st.warning(f"Could not display {Path(path).name}")
                current = corrections.get(path, predictions[path])
                st.selectbox(
                    Path(path).name, classes,
                    index=classes.index(current) if current in classes else 0,
                    key=f"sortimg_{path}",
                    on_change=_record_correction, args=(path,),
                )

    return {p: corrections.get(p, predictions[p]) for p in paths}


# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="ImageSorter", layout="wide")
st.title("ImageSorter")

tab_search, tab_train, tab_sort, tab_help = st.tabs(
    ["Search", "Train Custom Model", "Sort Images", "Help"]
)


# ================================================================
# TAB 1: SEARCH
# ================================================================
with tab_search:
    st.header("Search Images by Description")
    st.caption("Use natural language — e.g. *'sunset over water'*, *'my cat Chester'*, *'birthday party'*")

    search_folder = st.text_input("Folder to search", placeholder="C:/Users/you/Pictures")

    if search_folder:
        search_folder = search_folder.strip('"')
        if not os.path.isdir(search_folder):
            st.warning("Folder not found.")
        else:
            cache_key = f"emb_{search_folder}"
            if cache_key not in st.session_state:
                slot = st.empty()
                tagger = get_tagger(slot)
                bar = slot.progress(0.0, text="Indexing images with CLIP (once per folder)...")
                st.session_state[cache_key] = tagger.index_folder(
                    search_folder,
                    progress_callback=lambda done, total: bar.progress(
                        done / total, text=f"Indexing images with CLIP... {done} / {total}"
                    ),
                )
                slot.empty()

            embeddings = st.session_state[cache_key]
            st.success(f"Indexed {len(embeddings)} images.")

            query = st.text_input("Search for...", placeholder="e.g. 'a photo of mountains at sunset'")
            top_k = st.slider("Max results to show", min_value=5, max_value=50, value=20)

            if query and embeddings:
                results = st.session_state["tagger"].search(query, embeddings, top_k=top_k)
                cols_per_row = 4
                for i in range(0, len(results), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j, (path, score) in enumerate(results[i : i + cols_per_row]):
                        with cols[j]:
                            try:
                                st.image(load_thumbnail(path), use_container_width=True)
                                st.caption(f"{Path(path).name}  (score: {score:.3f})")
                            except Exception:
                                st.warning(f"Could not display {Path(path).name}")


# ================================================================
# TAB 2: TRAIN CUSTOM MODEL
# ================================================================
with tab_train:
    st.header("Train a Custom Sorting Model")
    st.markdown(
        "Organize a sample of your images into labeled subfolders, split across Train / "
        "Validation / Test sets. Each subfolder name becomes a category."
    )

    with st.expander("Don't have train / val / test folders yet? Split a sorted folder"):
        st.caption(
            "Pick a folder with one subfolder per category. Images are randomly copied "
            "(originals untouched) into `train`, `val` and `test` folders inside the output folder."
        )
        split_source = st.text_input("Folder of sorted images (one subfolder per category)", key="split_source")
        split_dest = st.text_input("Output folder (must be empty or new)", key="split_dest")
        sp1, sp2 = st.columns(2)
        with sp1:
            val_pct = st.slider("Validation %", 0, 40, 15)
        with sp2:
            test_pct = st.slider("Test %", 0, 40, 15)
        st.caption(f"Train gets the remaining {100 - val_pct - test_pct}%.")
        if st.button("Split Folder"):
            src, dst = split_source.strip('"'), split_dest.strip('"')
            if not src or not dst:
                st.error("Provide both folders.")
            elif val_pct + test_pct >= 100:
                st.error("Validation + test must be under 100%.")
            else:
                try:
                    ratios = ((100 - val_pct - test_pct) / 100, val_pct / 100, test_pct / 100)
                    counts = split_dataset(src, dst, ratios)
                    st.session_state["train_folder"] = str(Path(dst) / "train")
                    st.session_state["val_folder"] = str(Path(dst) / "val")
                    st.session_state["test_folder"] = str(Path(dst) / "test")
                    st.success(f"Split done: {counts}. Folder fields below were filled in.")
                except Exception as e:
                    st.error(f"Split failed: {e}")

    col1, col2, col3 = st.columns(3)
    with col1:
        train_folder = st.text_input("Train folder", key="train_folder")
    with col2:
        val_folder = st.text_input("Validation folder", key="val_folder")
    with col3:
        test_folder = st.text_input("Test folder", key="test_folder")

    col_ep, col_lr = st.columns(2)
    with col_ep:
        num_epochs = st.number_input("Epochs", min_value=1, max_value=100, value=5)
    with col_lr:
        learning_rate = st.number_input(
            "Learning rate", min_value=1e-5, max_value=1e-1, value=1e-3, format="%.5f"
        )

    col_dir, col_name = st.columns([3, 1])
    with col_dir:
        model_dir = st.text_input(
            "Save model to folder (created if missing)", value=str(MODELS_DIR), key="model_dir"
        )
    with col_name:
        model_name = st.text_input("Model file name", value="model.pt", key="model_name")
    model_name = model_name.strip() or "model.pt"
    if not model_name.lower().endswith(".pt"):
        model_name += ".pt"
    model_save_path = Path(model_dir.strip().strip('"')) / model_name
    if model_save_path.exists():
        st.warning(f"`{model_save_path}` already exists and will be overwritten.")
    else:
        st.caption(f"Will be saved as `{model_save_path}`")

    if st.button("Train Model", type="primary"):
        folders = [train_folder.strip('"'), val_folder.strip('"'), test_folder.strip('"')]
        if not all(folders) or not all(os.path.isdir(f) for f in folders):
            st.error("Provide valid paths for all three folders.")
        else:
            progress_bar = st.progress(0.0, text="Loading datasets...")
            chart_placeholder = st.empty()
            log_placeholder = st.empty()
            chart_data: dict = {"Train Loss": [], "Val Loss": []}
            epoch_lines: list = []

            def on_batch(phase, epoch, total_epochs, batch, total_batches):
                # Within an epoch, training is ~80% of the work and validation ~20%.
                within = (batch / total_batches) * (0.8 if phase == "train" else 0.2)
                within += 0.0 if phase == "train" else 0.8
                label = "training" if phase == "train" else "validating"
                progress_bar.progress(
                    min(((epoch - 1) + within) / total_epochs, 1.0),
                    text=f"Epoch {epoch}/{total_epochs} — {label} batch {batch}/{total_batches}",
                )

            def on_epoch(epoch, total, train_loss, val_loss):
                chart_data["Train Loss"].append(train_loss)
                chart_data["Val Loss"].append(val_loss)
                epoch_lines.append(
                    f"Epoch {epoch}/{total}  train loss: {train_loss:.4f}  val loss: {val_loss:.4f}"
                )
                log_placeholder.code("\n".join(epoch_lines), language=None)
                chart_placeholder.line_chart(chart_data)

            try:
                log.info("Train requested: folders=%s epochs=%s lr=%s", folders, num_epochs, learning_rate)
                trainer = ImageClassifierTrainer(*folders)
                if len(trainer.train_dataset) == 0:
                    raise ValueError("The train folder contains no images.")
                progress_bar.progress(0.0, text=f"Training on {len(trainer.train_dataset)} images...")
                trainer.train(
                    num_epochs=int(num_epochs),
                    learning_rate=float(learning_rate),
                    progress_callback=on_epoch,
                    batch_callback=on_batch,
                )
                progress_bar.progress(1.0, text="Saving model...")
                trainer.save(str(model_save_path))
                st.session_state["trained_model"] = trainer.model
                st.session_state["trained_classes"] = trainer.classes
                st.session_state["last_saved_model"] = str(model_save_path.resolve())
                progress_bar.progress(1.0, text="Training complete")
                st.success(
                    f"Model saved to `{model_save_path.resolve()}`.  Categories: {trainer.classes}"
                )
            except Exception as e:
                log.exception("Training failed")
                st.error(f"Training failed: {e}")


# ================================================================
# TAB 3: SORT IMAGES
# ================================================================
with tab_sort:
    st.header("Sort Images with Your Trained Model")

    sort_source = st.text_input("Folder of unsorted images", key="sort_source")
    sort_dest = st.text_input(
        "Destination folder (copies placed here, originals untouched)", key="sort_dest"
    )

    # Model selection
    has_trained = "trained_model" in st.session_state
    _clip_opt = "Use Pretrained CLIP (zero-shot)"
    if has_trained:
        model_source = st.radio(
            "Model to use",
            ["Freshly trained (from Train tab)", _clip_opt, "Load from file"],
        )
    else:
        model_source = st.radio(
            "Model to use",
            [_clip_opt, "Load from file"],
        )

    clip_labels = ""
    if model_source == _clip_opt:
        clip_labels = st.text_input(
            "Category labels (comma-separated)",
            placeholder="e.g. Chester, Nature, Food, Birthday",
            key="clip_labels",
        )

    sort_model_path = ""
    if model_source == "Load from file":
        saved = [str(p) for p in sorted(MODELS_DIR.glob("*.pt"))] if MODELS_DIR.is_dir() else []
        last_saved = st.session_state.get("last_saved_model")
        if last_saved and last_saved not in saved:
            saved.insert(0, last_saved)
        other = "Other location..."
        choice = st.selectbox(f"Saved models (from `{MODELS_DIR}`)", saved + [other])
        if choice == other:
            sort_model_path = st.text_input("Path to .pt model file", key="sort_model_path")
        else:
            sort_model_path = choice

    if st.button("Run Model on Folder"):
        src = sort_source.strip('"')
        if not os.path.isdir(src):
            st.error("Source folder not found.")
        elif model_source == _clip_opt:
            classes = [lbl.strip() for lbl in clip_labels.split(",") if lbl.strip()]
            if not classes:
                st.error("Enter at least one category label.")
            else:
                slot = st.empty()
                tagger = get_tagger(slot)
                image_paths = list_images(src)
                if not image_paths:
                    st.error("No images found in source folder.")
                else:
                    log.info("Zero-shot sort: %d images, labels=%s", len(image_paths), classes)
                    predictions = {}
                    progress_bar = slot.progress(0.0, text="Classifying images with CLIP...")
                    for i, img_path in enumerate(image_paths):
                        try:
                            label, _ = tagger.classify(img_path, classes)
                            predictions[img_path] = label
                        except Exception as e:
                            log.warning("Skipping %s: %s", img_path, e)
                        progress_bar.progress(
                            (i + 1) / len(image_paths),
                            text=f"Classifying images with CLIP... {i + 1} / {len(image_paths)}",
                        )
                    slot.empty()
                    store_predictions(predictions, classes, sort_dest.strip('"'))
        else:
            if has_trained and model_source == "Freshly trained (from Train tab)":
                model = st.session_state["trained_model"]
                classes = st.session_state["trained_classes"]
            elif sort_model_path and os.path.isfile(sort_model_path.strip('"')):
                with st.spinner("Loading model..."):
                    model, classes = ImageClassifierTrainer.load_model(sort_model_path.strip('"'))
            else:
                st.error("No model available — train one or provide a .pt file path.")
                st.stop()

            transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
            ])
            classifier = ImageClassifier(src, classes, model=model, imagetransform=transform)

            progress_bar = st.progress(0.0, text="Running predictions...")
            classifier.get_predictions(
                progress_callback=lambda done, total: progress_bar.progress(
                    done / total, text=f"Running predictions... {done}/{total}"
                )
            )
            progress_bar.empty()

            store_predictions(classifier.predictions, classes, sort_dest.strip('"'))

    # Review predictions and confirm copy
    if "sort_predictions" in st.session_state:
        st.markdown(f"**{len(st.session_state['sort_predictions'])} images** — review below and correct any mistakes.")
        confirmed = render_sort_preview(
            st.session_state["sort_predictions"],
            st.session_state["sort_classes"],
        )

        action = st.radio(
            "What should happen to the images?",
            ["Copy (originals stay where they are)", "Move (originals are removed from the source folder)"],
            key="sort_action",
        )
        do_move = action.startswith("Move")
        move_ok = True
        if do_move:
            st.warning(
                "Moving takes the images out of the source folder and puts them in the destination. "
                "Each image is copied and checked first, and the original is only deleted once the "
                "copy is confirmed good. This can't be undone from the app."
            )
            move_ok = st.checkbox("I understand — move my images", key="sort_move_ok")

        deep_verify = st.checkbox(
            "Deep verify (compare file contents, slower)",
            key="sort_deep_verify",
            help="By default each copy is verified by file size. Deep verify also compares a "
                 "checksum of the original and the copy, which reads every photo twice.",
        )

        verb, gerund, past = ("Move", "Moving", "moved") if do_move else ("Copy", "Copying", "copied")
        if st.button(
            f"{verb} Images to Destination", type="primary", disabled=not confirmed or not move_ok
        ):
            dest = st.session_state.get("sort_dest_confirmed") or sort_dest.strip('"')
            if not dest:
                st.error("Set a destination folder first.")
            else:
                progress_bar = st.progress(0.0, text=f"{gerund} and verifying images...")
                result = sort_images_to_folders(
                    confirmed, dest, move=do_move, deep_verify=deep_verify,
                    progress_callback=lambda done, total: progress_bar.progress(
                        done / total, text=f"{gerund} and verifying images... {done} / {total}"
                    ),
                )
                progress_bar.empty()
                log.info("UI %s: %d of %d to %s", past, result["done"], result["total"], dest)

                check = "contents compared" if result["deep"] else "sizes checked"
                if result["failed"] or result["problems"] or result["original_kept"]:
                    lines = [
                        f"{result['done']} of {result['total']} images {past} to `{dest}` "
                        f"({result['verified']} verified, {check})."
                    ]
                    if result["failed"]:
                        lines.append(
                            f"**{len(result['failed'])} failed** (originals left untouched):"
                        )
                        lines += [f"- `{Path(p).name}`: {err}" for p, err in result["failed"][:20]]
                    if result["original_kept"]:
                        lines.append(
                            f"**{len(result['original_kept'])} copied and verified, but the original "
                            "could not be removed:**"
                        )
                        lines += [f"- `{Path(p).name}`: {err}" for p, err in result["original_kept"][:20]]
                    if result["problems"]:
                        lines.append(
                            f"**{len(result['problems'])} failed the final check in the destination:**"
                        )
                        lines += [f"- `{Path(p).name}`: {err}" for p, err in result["problems"][:20]]
                    st.warning("\n\n".join(lines))
                else:
                    st.success(
                        f"Done! All {result['total']} images {past} to `{dest}` and verified ({check})."
                    )
                if result["manifest"]:
                    st.caption(f"Record of every file saved to `{result['manifest']}`")
                # Those paths are consumed (moved) or finished (copied); clear the review.
                del st.session_state["sort_predictions"]
                st.session_state["sort_corrections"] = {}


# ================================================================
# TAB 4: HELP
# ================================================================
with tab_help:
    if HELP_FILE.is_file():
        st.markdown(HELP_FILE.read_text(encoding="utf-8"))
    else:
        st.warning("HELP.md not found.")

    with st.expander("Debug log (most recent 200 lines)"):
        st.caption(f"Full log file: `{LOG_FILE}`")
        st.code(read_log_tail() or "(log is empty)", language=None)
