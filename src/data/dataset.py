"""Dataset loading, splitting, and preprocessing."""
import argparse
from pathlib import Path
import shutil
import random

def split_dataset(raw_dir: str, split_dir: str, ratios: tuple = (0.8, 0.1, 0.1)):
    """Split images into train/val/test folders."""
    raw = Path(raw_dir)
    splits = Path(split_dir)
    splits.mkdir(parents=True, exist_ok=True)
    
    for class_dir in raw.iterdir():
        if not class_dir.is_dir(): continue
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        random.shuffle(images)
        
        n = len(images)
        train_end = int(n * ratios[0])
        val_end = train_end + int(n * ratios[1])
        
        for folder, imgs in [
            ("train", images[:train_end]),
            ("val", images[train_end:val_end]),
            ("test", images[val_end:])
        ]:
            target = splits / folder / class_dir.name
            target.mkdir(parents=True, exist_ok=True)
            for img in imgs:
                shutil.copy(img, target / img.name)
    print(f"✅ Split complete. Output: {split_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", action="store_true", help="Run dataset splitting")
    args = parser.parse_args()
    if args.split:
        split_dataset("data/raw/tomato", "data/splits")
