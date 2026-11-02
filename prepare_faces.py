
import os
import cv2
import mediapipe as mp
from PIL import Image, ImageOps
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "face_landmarker.task"
IMAGE_SIZE = (224, 224)

DATASETS = {
    "train": "dataset_stratified/train",
    "validation": "dataset_stratified/validation",
    "test": "dataset_clean/testing_set"
}

OUTPUT_ROOT = "dataset_faces"
CLASSES = ["Heart", "Oblong", "Oval", "Round", "Square"]

# Initialize MediaPipe
base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1
)

landmarker = vision.FaceLandmarker.create_from_options(options)

total_processed = 0
failed_images = []

try:
    for split, input_folder in DATASETS.items():

        print(f"\nProcessing {split} dataset...")

        for class_name in CLASSES:

            source_folder = os.path.join(input_folder, class_name)
            output_folder = os.path.join(
                OUTPUT_ROOT, split, class_name
            )

            os.makedirs(output_folder, exist_ok=True)

            for filename in os.listdir(source_folder):

                if not filename.lower().endswith(
                    (".jpg", ".jpeg", ".png")
                ):
                    continue

                image_path = os.path.join(source_folder, filename)

                try:
                    image = cv2.imread(image_path)

                    if image is None:
                        failed_images.append(image_path)
                        continue

                    height, width = image.shape[:2]

                    rgb_image = cv2.cvtColor(
                        image, cv2.COLOR_BGR2RGB
                    )

                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=rgb_image
                    )

                    result = landmarker.detect(mp_image)

                    if not result.face_landmarks:
                        failed_images.append(image_path)
                        continue

                    landmarks = result.face_landmarks[0]

                    x_values = [p.x for p in landmarks]
                    y_values = [p.y for p in landmarks]

                    x_min = max(0, int(min(x_values) * width))
                    x_max = min(width, int(max(x_values) * width))
                    y_min = max(0, int(min(y_values) * height))
                    y_max = min(height, int(max(y_values) * height))

                    pad_x = int((x_max - x_min) * 0.20)
                    pad_y = int((y_max - y_min) * 0.20)

                    x_min = max(0, x_min - pad_x)
                    x_max = min(width, x_max + pad_x)
                    y_min = max(0, y_min - pad_y)
                    y_max = min(height, y_max + pad_y)

                    face = rgb_image[y_min:y_max, x_min:x_max]

                    if face.size == 0:
                        failed_images.append(image_path)
                        continue

                    face_image = Image.fromarray(face)
                    face_image = ImageOps.pad(
                            face_image,
                            IMAGE_SIZE,
                            method=Image.Resampling.LANCZOS,
                            color=(127, 127, 127),
                            centering=(0.5, 0.5)
                    )

                    output_path = os.path.join(
                        output_folder, filename
                    )

                    face_image.save(output_path)
                    total_processed += 1

                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    failed_images.append(image_path)

            print(f"{class_name}: completed")

finally:
    landmarker.close()

print("\n========== COMPLETED ==========")
print("Successfully processed:", total_processed)
print("Failed/no face detected:", len(failed_images))

with open("failed_face_images.txt", "w") as file:
    for path in failed_images:
        file.write(path + "\n")

print("Failed image list saved.")
print("Output folder:", OUTPUT_ROOT)
