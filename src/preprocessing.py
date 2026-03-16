"""
PneumoScan — Dataset Preprocessing
Reorganizes the Kaggle Chest X-Ray dataset from 2-class (Normal/Pneumonia)
to 3-class (Normal/Bacteria/Virus) based on filename prefixes.
"""

import os
import shutil
import config
from config import CLASS_NAMES


def reorganize_dataset(source_dir=None):
    """
    Reorganize dataset from:
        train/NORMAL/, train/PNEUMONIA/
    To:
        train/NORMAL/, train/BACTERIA/, train/VIRUS/

    Pneumonia images have filenames like:
        person1_bacteria_1.jpeg → BACTERIA
        person1_virus_1.jpeg   → VIRUS
    """
    if source_dir is None:
        source_dir = config.RAW_DATA_DIR

    splits = ["train", "val", "test"]
    stats = {}

    for split in splits:
        split_dir = os.path.join(source_dir, split)
        pneumonia_dir = os.path.join(split_dir, "PNEUMONIA")
        bacteria_dir = os.path.join(split_dir, "BACTERIA")
        virus_dir = os.path.join(split_dir, "VIRUS")

        # Create target directories
        os.makedirs(bacteria_dir, exist_ok=True)
        os.makedirs(virus_dir, exist_ok=True)

        counts = {"NORMAL": 0, "BACTERIA": 0, "VIRUS": 0}

        # Count existing NORMAL images
        normal_dir = os.path.join(split_dir, "NORMAL")
        if os.path.exists(normal_dir):
            counts["NORMAL"] = len([
                f for f in os.listdir(normal_dir)
                if f.lower().endswith((".jpeg", ".jpg", ".png"))
            ])

        # Sort pneumonia images by filename prefix
        if os.path.exists(pneumonia_dir):
            for filename in os.listdir(pneumonia_dir):
                if not filename.lower().endswith((".jpeg", ".jpg", ".png")):
                    continue

                src_path = os.path.join(pneumonia_dir, filename)
                filename_lower = filename.lower()

                if "bacteria" in filename_lower:
                    dst_path = os.path.join(bacteria_dir, filename)
                    counts["BACTERIA"] += 1
                elif "virus" in filename_lower:
                    dst_path = os.path.join(virus_dir, filename)
                    counts["VIRUS"] += 1
                else:
                    # Unknown pneumonia type — skip
                    print(f"  Warning: Cannot classify {filename}, skipping.")
                    continue

                # Copy (not move) to preserve original dataset
                if not os.path.exists(dst_path):
                    shutil.copy2(src_path, dst_path)

        # Remove original PNEUMONIA folder so only 3 class dirs remain
        if os.path.exists(pneumonia_dir):
            shutil.rmtree(pneumonia_dir)
            print(f"  Removed original PNEUMONIA/ folder from {split}")

        stats[split] = counts
        total = sum(counts.values())
        print(f"\n{split.upper()} split:")
        for cls, count in counts.items():
            pct = (count / total * 100) if total > 0 else 0
            print(f"  {cls}: {count} ({pct:.1f}%)")
        print(f"  Total: {total}")

    return stats


def merge_val_into_train(source_dir=None):
    """
    Merge the tiny validation set (24 images) into the training set.
    We'll use stratified K-Fold CV instead.
    """
    if source_dir is None:
        source_dir = config.RAW_DATA_DIR

    val_dir = os.path.join(source_dir, "val")
    train_dir = os.path.join(source_dir, "train")
    moved = 0

    for class_name in CLASS_NAMES:
        val_class_dir = os.path.join(val_dir, class_name)
        train_class_dir = os.path.join(train_dir, class_name)

        if not os.path.exists(val_class_dir):
            continue

        os.makedirs(train_class_dir, exist_ok=True)

        for filename in os.listdir(val_class_dir):
            if not filename.lower().endswith((".jpeg", ".jpg", ".png")):
                continue

            src = os.path.join(val_class_dir, filename)
            dst = os.path.join(train_class_dir, filename)

            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                moved += 1

    print(f"\nMerged {moved} validation images into training set.")
    return moved


def get_dataset_stats(source_dir=None):
    """Get image counts per class per split."""
    if source_dir is None:
        source_dir = config.RAW_DATA_DIR

    stats = {}
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(source_dir, split)
        stats[split] = {}
        for class_name in CLASS_NAMES:
            class_dir = os.path.join(split_dir, class_name)
            if os.path.exists(class_dir):
                count = len([
                    f for f in os.listdir(class_dir)
                    if f.lower().endswith((".jpeg", ".jpg", ".png"))
                ])
            else:
                count = 0
            stats[split][class_name] = count
    return stats


if __name__ == "__main__":
    print("=" * 50)
    print("PneumoScan — Dataset Preprocessing")
    print("=" * 50)

    print("\nStep 1: Reorganizing to 3-class structure...")
    reorganize_dataset()

    print("\nStep 2: Merging validation into training set...")
    merge_val_into_train()

    print("\nFinal dataset statistics:")
    stats = get_dataset_stats()
    for split, counts in stats.items():
        print(f"\n{split.upper()}:")
        for cls, count in counts.items():
            print(f"  {cls}: {count}")

    print("\nDone!")
