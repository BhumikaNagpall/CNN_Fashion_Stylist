
import os

root = "dataset_faces_clean"

splits = ["train", "validation", "test"]
classes = ["Heart", "Oblong", "Oval", "Round", "Square"]

total = 0

for split in splits:
    print(f"\n{split.upper()}")

    for class_name in classes:
        folder = os.path.join(root, split, class_name)

        count = len([
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        total += count
        print(f"{class_name}: {count}")

print("\nTotal unique images:", total)
