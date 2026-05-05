import os
from typing import Optional

COMMON_IMAGE_FOLDERS = ["train", "test", "validation", "val", "images", ""]

def resolve_image_path(image_root: str, image_name: str, split: Optional[str] = None, image_folder: Optional[str] = None) -> str:
    """Resolve image path for datasets that keep images in split folders.

    Priority:
    1) absolute path in CSV
    2) CSV column image_folder, e.g. train/test
    3) CSV split column, e.g. train/test/val
    4) fallback search in train/test/validation/val/images/root
    """
    image_name = str(image_name)
    if os.path.isabs(image_name) and os.path.exists(image_name):
        return image_name

    candidates = []
    if image_folder and str(image_folder).strip() and str(image_folder).lower() != "nan":
        candidates.append(os.path.join(image_root, str(image_folder), image_name))
    if split and str(split).strip() and str(split).lower() != "nan":
        split = str(split).lower()
        # val rows usually come from train images when auto-split from train.csv
        candidates.append(os.path.join(image_root, split, image_name))
        if split in {"val", "validation"}:
            candidates.append(os.path.join(image_root, "train", image_name))
    for folder in COMMON_IMAGE_FOLDERS:
        candidates.append(os.path.join(image_root, folder, image_name) if folder else os.path.join(image_root, image_name))

    seen = set()
    for path in candidates:
        if path not in seen and os.path.exists(path):
            return path
        seen.add(path)
    raise FileNotFoundError(
        f"Không tìm thấy ảnh {image_name}. Đã thử trong: {image_root}/train, {image_root}/test, {image_root}/validation, {image_root}/val, {image_root}/images"
    )
