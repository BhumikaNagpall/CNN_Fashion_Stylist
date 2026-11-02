
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Load independent test dataset
test_dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset_faces_clean/test",
    image_size=(224, 224),
    batch_size=16,
    label_mode="categorical",
    shuffle=False
)

class_names = test_dataset.class_names

# Load fine-tuned model
model = tf.keras.models.load_model(
    "best_mobilenet_finetuned.keras"
)

# Actual labels
y_true = np.concatenate([
    labels.numpy().argmax(axis=1)
    for images, labels in test_dataset
])

# Predictions
predictions = model.predict(test_dataset)
y_pred = np.argmax(predictions, axis=1)

# Classification report
print("\nFine-tuned Model Classification Report:")

print(classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    zero_division=0
))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(cmap="Blues", xticks_rotation=45)
plt.title("Fine-tuned MobileNetV2 Confusion Matrix")
plt.tight_layout()
plt.show()
