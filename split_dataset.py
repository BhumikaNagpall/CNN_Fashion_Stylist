
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

SOURCE = Path("dataset_clean/training_set")
DESTINATION = Path("dataset_stratified")

CLASSES = ["Heart", "Oblong", "Oval", "Round", "Square"]

for class_name in CLASSES:

    class_folder = SOURCE / class_name

    images = [
        p for p in class_folder.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    train_images, val_images = train_test_split(
        images,
        test_size=0.2,
        random_state=42
    )

    for split_name, image_list in [
        ("train", train_images),
        ("validation", val_images)
    ]:

        output_folder = DESTINATION / split_name / class_name
        output_folder.mkdir(parents=True, exist_ok=True)

        for image_path in image_list:
            shutil.copy2(
                image_path,
                output_folder / image_path.name
            )

    print(
        f"{class_name}: "
        f"Training={len(train_images)}, "
        f"Validation={len(val_images)}"
    )

print("\nStratified dataset created successfully!")
