import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

# Enable GPU if available
device = "GPU" if tf.config.experimental.list_physical_devices('GPU') else "CPU"
print(f"Using {device} for training")

# Define dataset path
DATA_DIR = 'dataset/'  # Ensure this directory contains class subfolders

# Check if dataset exists
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Dataset directory '{DATA_DIR}' not found!")

# Data augmentation & preprocessing
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,  
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.1  
)

# Define batch size dynamically
BATCH_SIZE = min(16, max(1, len(os.listdir(DATA_DIR)) // 5))

train_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Check dataset availability
if train_generator.samples == 0 or val_generator.samples == 0:
    raise ValueError("Training or validation dataset is empty. Check dataset structure.")

# Load pre-trained MobileNetV2 model (excluding top layers)
base_model = keras.applications.MobileNetV2(
    weights='imagenet', include_top=False, input_shape=(224, 224, 3)
)
base_model.trainable = False  # Freeze base model initially

# Build custom classification model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model_checkpoint = callbacks.ModelCheckpoint("best_model.h5", save_best_only=True, monitor='val_accuracy')

# Train model
history = model.fit(
    train_generator,
    epochs=15,
    validation_data=val_generator,
    callbacks=[early_stopping, model_checkpoint]
)

# Fine-tune base model (unfreeze last layers)
base_model.trainable = True
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history_finetune = model.fit(
    train_generator,
    epochs=5,
    validation_data=val_generator,
    callbacks=[early_stopping, model_checkpoint]
)

# Plot training history
def plot_history(histories):
    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    for history in histories:
        plt.plot(history.history['accuracy'], label=f"Train ({len(history.epoch)} epochs)")
        plt.plot(history.history['val_accuracy'], label=f"Val ({len(history.epoch)} epochs)")
    plt.legend()
    plt.title("Model Accuracy")

    # Loss
    plt.subplot(1, 2, 2)
    for history in histories:
        plt.plot(history.history['loss'], label=f"Train ({len(history.epoch)} epochs)")
        plt.plot(history.history['val_loss'], label=f"Val ({len(history.epoch)} epochs)")
    plt.legend()
    plt.title("Model Loss")

    plt.show()

plot_history([history, history_finetune])

# Save final model
model.save('final_image_classifier.h5')
print("Final model saved as 'final_image_classifier.h5'")
