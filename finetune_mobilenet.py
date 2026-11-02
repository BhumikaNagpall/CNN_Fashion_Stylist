
import tensorflow as tf

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 8

# Load datasets
train_dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset_faces_clean/train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True,
    seed=42
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset_faces_clean/validation",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(AUTOTUNE)
validation_dataset = validation_dataset.prefetch(AUTOTUNE)

# Load our previously trained model
model = tf.keras.models.load_model("best_mobilenet.keras")

# Find MobileNetV2 backbone
base_model = None

for layer in model.layers:
    if isinstance(layer, tf.keras.Model) and "mobilenetv2" in layer.name.lower():
        base_model = layer
        break

if base_model is None:
    raise ValueError("MobileNetV2 backbone not found in saved model.")

# Unfreeze backbone
base_model.trainable = True

# Freeze all except final 30 layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Keep BatchNormalization layers frozen for stable fine-tuning
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

# Recompile with a very low learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00001
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Save best fine-tuned checkpoint
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "best_mobilenet_finetuned.keras",
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1
)

# Continue training
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stopping]
)

print("\nFine-tuning completed!")
print("Best checkpoint: best_mobilenet_finetuned.keras")
