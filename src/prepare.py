
import os
import shutil
import random
from glob import glob

random.seed(42)

raw_dir = "data/raw"
out_dir = "data/raw"

splits = ["normal", "cancer"]
split_ratio = [0.7, 0.2, 0.1]  

classes = ["cancer", "normal"]

for split in splits:
    for cls in classes:
        os.makedirs(os.path.join(out_dir, split, cls), exist_ok=True)

for cls in classes:
    images = glob(os.path.join(raw_dir, cls, "*"))
    random.shuffle(images)

    train_end = int(split_ratio[0] * len(images))
    val_end = train_end + int(split_ratio[1] * len(images))

    split_data = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split, imgs in split_data.items():
        for img in imgs:
            shutil.copy(img, os.path.join(out_dir, split, cls))
print("✅ Dataset split into train/val/test")