
from pathlib import Path
from PIL import Image
import shutil

SOURCE = Path("dataset")
DESTINATION = Path("dataset_clean")

valid_extensions = {".jpg", ".jpeg", ".png"}

copied = 0
skipped = 0

for image_path in SOURCE.rglob("*"):

    if not image_path.is_file():
        continue

    if image_path.suffix.lower() not in valid_extensions:
        continue

    try:
        with Image.open(image_path) as img:
            img.verify()

        with Image.open(image_path) as img:
            img.load()

        # Preserve original folder structure
        relative_path = image_path.relative_to(SOURCE)
        destination_path = DESTINATION / relative_path

        destination_path.parent.mkdir(
            parents=True, exist_ok=True
        )

        shutil.copy2(image_path, destination_path)
        copied += 1

    except Exception as error:
        print("Skipping:", image_path)
        print("Reason:", error)
        skipped += 1

print("\n========== CLEANING COMPLETE ==========")
print("Valid images copied:", copied)
print("Corrupted images skipped:", skipped)
print("Clean dataset:", DESTINATION)
