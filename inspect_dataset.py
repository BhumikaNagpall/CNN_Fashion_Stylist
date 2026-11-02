
import os
import random
import hashlib
from PIL import Image
import matplotlib.pyplot as plt

# Dataset locations
folders = {
    "Training": "dataset_stratified/train",
    "Validation": "dataset_stratified/validation",
    "Testing": "dataset_clean/testing_set"
}

classes = ["Heart", "Oblong", "Oval", "Round", "Square"]

random.seed(42)

all_hashes = {}
total_images = 0

print("\n========== DATASET INSPECTION ==========")

for split, folder in folders.items():

    print(f"\n--- {split} DATASET ---")

    for class_name in classes:

        class_path = os.path.join(folder, class_name)

        files = [
            f for f in os.listdir(class_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        total_images += len(files)

        print(f"{class_name}: {len(files)} images")

        dimensions = []

        for filename in files:
            path = os.path.join(class_path, filename)

            try:
                with Image.open(path) as img:
                    dimensions.append(img.size)

                    # Check duplicate image content
                    image_hash = hashlib.md5(
                        img.convert("RGB").tobytes()
                    ).hexdigest()

                    all_hashes.setdefault(image_hash, []).append(
                        f"{split}/{class_name}/{filename}"
                    )

            except Exception as e:
                print("Problem with:", path, e)

        if dimensions:
            widths = [d[0] for d in dimensions]
            heights = [d[1] for d in dimensions]

            print(
                f"  Width range: {min(widths)}-{max(widths)}, "
                f"Height range: {min(heights)}-{max(heights)}"
            )

# Duplicate analysis
duplicates = {
    key: value
    for key, value in all_hashes.items()
    if len(value) > 1
}

print("\n========== SUMMARY ==========")
print("Total images:", total_images)
print("Duplicate image groups:", len(duplicates))

for group in list(duplicates.values())[:10]:
    print("Duplicate:", group)

# Visual inspection: 5 random images per class
fig, axes = plt.subplots(5, 5, figsize=(12, 12))

for row, class_name in enumerate(classes):

    class_path = os.path.join("dataset_stratified/train", class_name)

    files = [
        f for f in os.listdir(class_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    samples = random.sample(files, min(5, len(files)))

    for col, filename in enumerate(samples):

        image_path = os.path.join(class_path, filename)

        with Image.open(image_path) as img:
            axes[row, col].imshow(img.convert("RGB"))

        axes[row, col].axis("off")

        if col == 0:
            axes[row, col].set_ylabel(class_name, fontsize=11)

plt.suptitle("Random Training Images by Face Shape")
plt.tight_layout()
plt.show()
