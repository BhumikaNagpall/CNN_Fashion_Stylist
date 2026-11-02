
import tensorflow as tf
import matplotlib.pyplot as plt

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

TRAIN_DIR = "dataset/training_set"
TEST_DIR = "dataset/testing_set"

# Load datasets
train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

class_names = train_dataset.class_names

# Data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])

# Normalize images
normalization = tf.keras.layers.Rescaling(1./255)

# Preprocess training data
train_dataset = train_dataset.map(
    lambda x, y: (normalization(data_augmentation(x, training=True)), y),
    num_parallel_calls=tf.data.AUTOTUNE
)

# Preprocess testing data (no augmentation)
test_dataset = test_dataset.map(
    lambda x, y: (normalization(x), y),
    num_parallel_calls=tf.data.AUTOTUNE
)

# Optimize performance
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

print("Preprocessing completed!")
print("Classes:", class_names)

images, labels = next(iter(train_dataset))

print("Processed image shape:", images.shape)
print("Minimum pixel value:", tf.reduce_min(images).numpy())
print("Maximum pixel value:", tf.reduce_max(images).numpy())

# Visualize samples
plt.figure(figsize=(10, 8))

for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy())
    plt.title(class_names[tf.argmax(labels[i]).numpy()])
    plt.axis("off")

plt.tight_layout()
plt.show()
