
import tensorflow as tf
import numpy as np

# Load trained model
model = tf.keras.models.load_model(
    "improved_face_shape_model.keras"
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Load validation data
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset_clean/training_set",
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

# Load independent test data
test_dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset_clean/testing_set",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

class_names = validation_dataset.class_names

def diagnose(dataset, name):
    true_labels = []
    predicted_labels = []

    for images, labels in dataset:
        predictions = model.predict(images, verbose=0)

        true_labels.extend(
            np.argmax(labels.numpy(), axis=1)
        )

        predicted_labels.extend(
            np.argmax(predictions, axis=1)
        )

    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    print(f"\n========== {name} ==========")

    print("\nActual class distribution:")
    print(np.bincount(true_labels, minlength=5))

    print("\nPredicted class distribution:")
    print(np.bincount(predicted_labels, minlength=5))

    matrix = tf.math.confusion_matrix(
        true_labels,
        predicted_labels,
        num_classes=5
    ).numpy()

    print("\nConfusion Matrix:")
    print(matrix)

    accuracy = np.mean(true_labels == predicted_labels)
    print("\nAccuracy:", accuracy)

    print("\nClass order:", class_names)


diagnose(validation_dataset, "VALIDATION RESULTS")
diagnose(test_dataset, "TEST RESULTS")
