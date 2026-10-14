import csv
import hashlib
import logging
import os
import re
import shutil
from datetime import datetime

from tqdm import tqdm

log = logging.getLogger("imagesorter.transfer")

_INVALID_FOLDER_CHARS = re.compile(r'[<>:"/\\|?*]')


def folder_name(label: str) -> str:
    """Make a category label safe to use as a folder name."""
    return _INVALID_FOLDER_CHARS.sub("_", label).strip(" .") or "_"


def unique_path(folder: str, filename: str) -> str:
    """Return folder/filename, or folder/name (1).ext etc. if that already exists."""
    stem, ext = os.path.splitext(filename)
    candidate = os.path.join(folder, filename)
    n = 1
    while os.path.exists(candidate):
        candidate = os.path.join(folder, f"{stem} ({n}){ext}")
        n += 1
    return candidate


def file_sha256(path: str) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _check_copy(src: str, dst: str, expected_size: int, deep: bool):
    """
    Verify that dst is a faithful copy of src. Returns (error_or_None, sha256_or_"").
    The quick check compares the file size; deep=True also compares SHA-256 hashes.
    """
    if not os.path.isfile(dst):
        return "missing at destination", ""
    actual = os.path.getsize(dst)
    if actual != expected_size:
        return f"size mismatch ({actual} vs {expected_size} bytes)", ""
    if deep:
        src_hash, dst_hash = file_sha256(src), file_sha256(dst)
        if src_hash != dst_hash:
            return "content differs from the original", ""
        return None, dst_hash
    return None, ""


def sort_images_to_folders(
    confirmed_predictions: dict,
    dest_folder: str,
    move: bool = False,
    progress_callback=None,
    deep_verify: bool = False,
    write_manifest: bool = True,
) -> dict:
    """
    Copy (default) or move images into dest_folder/<label>/ from an
    {image_path: label} dict, verifying every file.

    Every image goes through the same steps: copy -> verify -> (move only)
    delete the original. A failed or unverifiable copy is removed and the
    original is left exactly where it was, so a bad transfer can never lose a
    photo. One failing file does not abort the run. Existing files are never
    overwritten: a name clash becomes "name (1).ext".

    deep_verify=True also compares SHA-256 hashes of the original and the copy
    (slower; reads every file twice). Otherwise file sizes are compared.

    After the run, every destination file is checked once more and (unless
    write_manifest=False) a CSV listing each image's original path, new path,
    label, size and status is written to dest_folder/sort_manifest_<time>.csv.

    progress_callback(done, total) is called after every image.
    Returns {
        "mode", "total", "done", "verified", "deep",
        "failed":        [(source_path, message), ...],   # nothing transferred
        "original_kept": [(source_path, message), ...],   # move: copy verified but original couldn't be removed
        "problems":      [(destination_path, message), ...],  # failed the final re-check
        "manifest":      path or None,
    }
    """
    mode = "move" if move else "copy"
    total = len(confirmed_predictions)
    log.info("Starting %s of %d images into %s (deep verify: %s)", mode, total, dest_folder, deep_verify)

    records, failed, original_kept = [], [], []
    done = 0
    for i, (src, label) in enumerate(
        tqdm(confirmed_predictions.items(), desc=f"{mode.capitalize()} images"), start=1
    ):
        rec = {"mode": mode, "label": label, "source": src, "destination": "",
               "size_bytes": "", "sha256": "", "status": ""}
        records.append(rec)
        dst, created = None, False
        try:
            if not os.path.isfile(src):
                raise FileNotFoundError("source file no longer exists")
            size = os.path.getsize(src)
            rec["size_bytes"] = size

            dest_dir = os.path.join(dest_folder, folder_name(label))
            os.makedirs(dest_dir, exist_ok=True)
            dst = unique_path(dest_dir, os.path.basename(src))
            rec["destination"] = dst

            created = True  # dst did not exist a moment ago, so we own it
            shutil.copy2(src, dst)
            error, digest = _check_copy(src, dst, size, deep_verify)
            if error:
                raise RuntimeError(f"copy failed verification: {error}")
            rec["sha256"] = digest

            if move:
                try:
                    os.remove(src)
                    rec["status"] = "verified; original removed"
                except OSError as e:
                    rec["status"] = "verified; original NOT removed"
                    original_kept.append((src, f"copied and verified, but could not remove the original: {e}"))
                    log.warning("Could not remove original %s: %s", src, e)
            else:
                rec["status"] = "verified"
            done += 1
        except Exception as e:
            if created and dst and os.path.exists(dst):
                try:
                    os.remove(dst)  # never leave a partial/bad copy behind
                except OSError:
                    pass
            rec["status"] = f"FAILED: {e}"
            failed.append((src, str(e)))
            log.error("Failed to %s %s: %s", mode, src, e)
        if progress_callback:
            progress_callback(i, total)

    # Final pass: confirm everything reported as transferred is really there.
    verified, problems = 0, []
    for rec in records:
        if rec["status"].startswith("FAILED"):
            continue
        dst = rec["destination"]
        if not os.path.isfile(dst):
            msg = "missing at destination"
        elif os.path.getsize(dst) != rec["size_bytes"]:
            msg = "size changed after transfer"
        else:
            verified += 1
            continue
        rec["status"] = f"PROBLEM: {msg}"
        problems.append((dst, msg))
        log.error("Final check failed for %s: %s", dst, msg)

    manifest = None
    if write_manifest and records:
        try:
            os.makedirs(dest_folder, exist_ok=True)
            manifest = os.path.join(
                dest_folder, f"sort_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            with open(manifest, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["mode", "label", "source", "destination", "size_bytes", "sha256", "status"]
                )
                writer.writeheader()
                writer.writerows(records)
            log.info("Manifest written to %s", manifest)
        except OSError as e:
            manifest = None
            log.error("Could not write manifest: %s", e)

    log.info(
        "%s finished: %d done, %d verified, %d failed, %d originals kept, %d problems",
        mode.capitalize(), done, verified, len(failed), len(original_kept), len(problems),
    )
    return {
        "mode": mode, "total": total, "done": done, "verified": verified, "deep": deep_verify,
        "failed": failed, "original_kept": original_kept, "problems": problems, "manifest": manifest,
    }


def copy_images_to_folders(confirmed_predictions: dict, dest_folder: str, **kwargs) -> dict:
    """Copy images to dest_folder/<label>/ (originals untouched). See sort_images_to_folders."""
    return sort_images_to_folders(confirmed_predictions, dest_folder, move=False, **kwargs)


def move_images_to_folders(confirmed_predictions: dict, dest_folder: str, **kwargs) -> dict:
    """Move images to dest_folder/<label>/ (copy, verify, then delete the original)."""
    return sort_images_to_folders(confirmed_predictions, dest_folder, move=True, **kwargs)
