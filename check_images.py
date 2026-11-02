
from pathlib import Path
from PIL import Image

DATASET_DIR = Path("dataset")

bad_images = []
total_images = 0

valid_extensions = {".jpg", ".jpeg", ".png"}

for image_path in DATASET_DIR.rglob("*"):

    if image_path.suffix.lower() not in valid_extensions:
        continue

    total_images += 1

    try:
        # Verify image integrity
        with Image.open(image_path) as img:
            img.verify()

        # Reopen and fully decode the image
        with Image.open(image_path) as img:
            img.load()

    except Exception as error:
        bad_images.append(image_path)
        print("Corrupted image:", image_path)
        print("Reason:", error)

print("\n========== SCAN RESULTS ==========")
print("Total images checked:", total_images)
print("Valid images:", total_images - len(bad_images))
print("Corrupted images:", len(bad_images))

# Save report
with open("bad_images.txt", "w") as file:
    for path in bad_images:
        file.write(str(path) + "\n")

print("\nReport saved to bad_images.txt")
