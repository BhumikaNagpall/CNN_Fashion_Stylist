
import os
import shutil
import hashlib

SOURCE = "dataset_faces"
DESTINATION = "dataset_faces_clean"

SPLITS = ["train", "validation", "test"]
CLASSES = ["Heart", "Oblong", "Oval", "Round", "Square"]

# Track image hashes across all splits
seen_hashes = set()

total_copied = 0
total_duplicates = 0

for split in SPLITS:
    print(f"\nProcessing {split}...")

    for class_name in CLASSES:

        source_folder = os.path.join(SOURCE, split, class_name)
        destination_folder = os.path.join(
            DESTINATION, split, class_name
        )

        os.makedirs(destination_folder, exist_ok=True)

        for filename in sorted(os.listdir(source_folder)):

            source_path = os.path.join(source_folder, filename)

            if not filename.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):
                continue

            try:
                with open(source_path, "rb") as file:
                    file_hash = hashlib.md5(file.read()).hexdigest()

                if file_hash in seen_hashes:
                    total_duplicates += 1
                    print("Duplicate skipped:", split, class_name, filename)
                    continue

                seen_hashes.add(file_hash)

                destination_path = os.path.join(
                    destination_folder, filename
                )

                shutil.copy2(source_path, destination_path)
                total_copied += 1

            except Exception as e:
                print("Error:", source_path, e)

print("\n========== DUPLICATE CLEANING COMPLETE ==========")
print("Images copied:", total_copied)
print("Duplicates skipped:", total_duplicates)
print("Output:", DESTINATION)
