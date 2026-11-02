
import os
import random
import matplotlib.pyplot as plt
from PIL import Image

classes = ["Heart", "Oblong", "Oval", "Round", "Square"]

random.seed(42)

fig, axes = plt.subplots(5, 5, figsize=(12, 12))

for row, class_name in enumerate(classes):

    folder = os.path.join("dataset_faces", "train", class_name)

    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    samples = random.sample(files, 5)

    for col, filename in enumerate(samples):

        path = os.path.join(folder, filename)

        with Image.open(path) as img:
            axes[row, col].imshow(img.convert("RGB"))

        axes[row, col].axis("off")

        if col == 0:
            axes[row, col].set_ylabel(
                class_name, fontsize=12
            )

plt.suptitle("Face-Cropped Dataset Inspection")
plt.tight_layout()
plt.show()
