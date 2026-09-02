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

sys.path.append(str(Path(__file__).parent / "ImageClassifier Files"))
from ImageClassifierFiles.ImageClassifier import ImageClassifier, copy_images_to_folders
from ImageClassifierFiles.ImageClassifierTrainer import ImageClassifierTrainer
from ImageClassifierFiles.PretrainedTagger import PretrainedTagger


# ── Sort preview (YOUR CONTRIBUTION) ────────────────────────────────────────

def render_sort_preview(predictions: dict, classes: list) -> dict:
    """
    Display the model's sorting predictions and let the user correct mistakes
    before committing the copy.

    Args:
        predictions: dict mapping image_path (str) -> predicted_label (str)
        classes:     list of all available category labels (e.g. ["Chester", "Nature"])

    Returns:
        dict mapping image_path -> confirmed_label
        (return {} if there's nothing to show yet)

    This is called on every Streamlit rerun, so the returned dict always
    reflects the current widget values — no separate "save" step needed.
    The "Copy Images" button that triggers the actual file copy lives outside
    this function, so you only need to display and collect corrections here.

    TODO: Implement this (~5-10 lines). Consider:
      - Show each image in a grid using st.columns(N)
      - Use st.selectbox to let the user correct a wrong label
      - Use key=f"sort_{path}" on each widget so Streamlit tracks them individually
      - Optionally group images by predicted category first

    Starter template:
        cols_per_row = 4
        col_list = st.columns(cols_per_row)
        for i, (path, pred_label) in enumerate(predictions.items()):
            with col_list[i % cols_per_row]:
                st.image(path, use_container_width=True)
                confirmed[path] = st.selectbox(
                    Path(path).name, classes,
                    index=classes.index(pred_label) if pred_label in classes else 0,
                    key=f"sort_{path}",
                )
    """
    confirmed = {}
    cols_per_row = 4
    col_list = st.columns(cols_per_row)
    for i, (path, pred_label) in enumerate(predictions.items()):
        with col_list[i % cols_per_row]:
            st.image(path, use_container_width=True)
            confirmed[path] = st.selectbox(
                    Path(path).name, classes,
                    index=classes.index(pred_label) if pred_label in classes else 0,
                    key=f"sort_{path}",
                )
    # YOUR CODE HERE
    return confirmed


# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="ImageSorter", layout="wide")
st.title("ImageSorter")

tab_search, tab_train, tab_sort = st.tabs(["Search", "Train Custom Model", "Sort Images"])


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
                with st.spinner("Indexing images with CLIP (only done once per folder)..."):
                    tagger = PretrainedTagger()
                    st.session_state[cache_key] = tagger.index_folder(search_folder)
                    st.session_state["tagger"] = tagger

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
                                st.image(path, use_container_width=True)
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

    model_save_path = st.text_input("Save model to (.pt file)", value="model.pt")

    if st.button("Train Model", type="primary"):
        folders = [train_folder.strip('"'), val_folder.strip('"'), test_folder.strip('"')]
        if not all(folders) or not all(os.path.isdir(f) for f in folders):
            st.error("Provide valid paths for all three folders.")
        else:
            progress_bar = st.progress(0.0)
            status = st.empty()
            chart_placeholder = st.empty()
            chart_data: dict = {"Train Loss": [], "Val Loss": []}

            def on_epoch(epoch, total, train_loss, val_loss):
                chart_data["Train Loss"].append(train_loss)
                chart_data["Val Loss"].append(val_loss)
                progress_bar.progress(epoch / total)
                status.text(
                    f"Epoch {epoch}/{total}  —  train: {train_loss:.4f}  val: {val_loss:.4f}"
                )
                chart_placeholder.line_chart(chart_data)

            try:
                trainer = ImageClassifierTrainer(*folders)
                trainer.train(
                    num_epochs=int(num_epochs),
                    learning_rate=float(learning_rate),
                    progress_callback=on_epoch,
                )
                trainer.save(model_save_path)
                st.session_state["trained_model"] = trainer.model
                st.session_state["trained_classes"] = trainer.classes
                progress_bar.progress(1.0)
                st.success(f"Saved to `{model_save_path}`.  Categories: {trainer.classes}")
            except Exception as e:
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
        st.info("No model trained this session — load a saved .pt file below.")
        sort_model_path = st.text_input("Path to .pt model file", key="sort_model_path")

    if st.button("Run Model on Folder"):
        src = sort_source.strip('"')
        if not os.path.isdir(src):
            st.error("Source folder not found.")
        elif model_source == _clip_opt:
            classes = [lbl.strip() for lbl in clip_labels.split(",") if lbl.strip()]
            if not classes:
                st.error("Enter at least one category label.")
            else:
                if "tagger" not in st.session_state:
                    with st.spinner("Loading CLIP model..."):
                        st.session_state["tagger"] = PretrainedTagger()
                tagger = st.session_state["tagger"]
                image_paths = [
                    str(p)
                    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
                    for p in Path(src).glob(f"*{ext}")
                ]
                if not image_paths:
                    st.error("No images found in source folder.")
                else:
                    predictions = {}
                    progress_bar = st.progress(0.0)
                    for i, img_path in enumerate(image_paths):
                        label, _ = tagger.classify(img_path, classes)
                        predictions[img_path] = label
                        progress_bar.progress((i + 1) / len(image_paths))
                    st.session_state["sort_predictions"] = predictions
                    st.session_state["sort_classes"] = classes
                    st.session_state["sort_dest"] = sort_dest.strip('"')
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

            with st.spinner("Loading images..."):
                classifier.loadImages()
            with st.spinner("Running predictions..."):
                classifier.get_predictions()

            st.session_state["sort_predictions"] = classifier.predictions
            st.session_state["sort_classes"] = classes
            st.session_state["sort_dest"] = sort_dest.strip('"')

    # Review predictions and confirm copy
    if "sort_predictions" in st.session_state:
        st.markdown(f"**{len(st.session_state['sort_predictions'])} images** — review below and correct any mistakes.")
        confirmed = render_sort_preview(
            st.session_state["sort_predictions"],
            st.session_state["sort_classes"],
        )

        if st.button("Copy Images to Destination", type="primary", disabled=not confirmed):
            dest = st.session_state.get("sort_dest") or sort_dest.strip('"')
            if not dest:
                st.error("Set a destination folder first.")
            else:
                with st.spinner("Copying images..."):
                    copy_images_to_folders(confirmed, dest)
                st.success(f"Done! {len(confirmed)} images copied to `{dest}`")
                del st.session_state["sort_predictions"]
