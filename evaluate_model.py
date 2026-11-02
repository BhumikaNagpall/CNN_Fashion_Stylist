
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# Dataset settings
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

TEST_DIR = "dataset_clean/testing_set"

# Load test dataset
test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

class_names = test_dataset.class_names

# Load already trained model
model = tf.keras.models.load_model("best_improved_cnn.keras")

# Get actual labels
y_true = np.concatenate([
    labels.numpy().argmax(axis=1)
    for images, labels in test_dataset
])

# Predict without training
predictions = model.predict(test_dataset)
y_pred = np.argmax(predictions, axis=1)

# Classification report
print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=class_names
))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Face Shape Classification")
plt.tight_layout()
plt.show()
