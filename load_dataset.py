
import tensorflow as tf
import matplotlib.pyplot as plt

# Dataset directories
TRAIN_DIR = "dataset/training_set"
TEST_DIR = "dataset/testing_set"

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Load training dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True
)

# Load testing dataset
test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

# Get class names
class_names = train_dataset.class_names

print("\nClasses:", class_names)
print("Training batches:", len(train_dataset))
print("Testing batches:", len(test_dataset))

# Display image and label shapes
images, labels = next(iter(train_dataset))

print("Image shape:", images.shape)
print("Label shape:", labels.shape)

# Display sample images
plt.figure(figsize=(10, 8))

for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[tf.argmax(labels[i]).numpy()])
    plt.axis("off")

plt.tight_layout()
plt.show()
