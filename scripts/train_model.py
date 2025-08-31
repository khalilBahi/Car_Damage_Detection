import os
import json
import random
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.efficientnet import EfficientNetB0, preprocess_input
from keras import layers, models, regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

# ------------------------------
# Reproducibility
# ------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ------------------------------
# Paths and constants
# ------------------------------
TRAIN_DIR = pathlib.Path('../data/train')
VAL_DIR = pathlib.Path('../data/val')
MODELS_DIR = pathlib.Path('../models')
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS_INITIAL = 15  # head training
EPOCHS_FINE = 20     # fine-tuning

# Ensure required directories exist
if not TRAIN_DIR.exists() or not VAL_DIR.exists():
    raise FileNotFoundError("Expected ../data/train and ../data/val to exist. Please create proper splits.")

MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------
# Data pipeline (EfficientNet preprocessing + milder augmentation)
# ------------------------------
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.9, 1.1]
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=SEED,
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
)

num_classes = len(train_generator.class_indices)
print(f"Classes: {train_generator.class_indices}")

# Persist class indices for consistent evaluation
with open(MODELS_DIR / 'class_indices.json', 'w') as f:
    json.dump(train_generator.class_indices, f, indent=2)

# ------------------------------
# Class weights (clipped to avoid extreme imbalance impact)
# ------------------------------
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes,
)
class_weights_dict = dict(enumerate(class_weights))
# Clip weights into a reasonable band
for k in list(class_weights_dict.keys()):
    class_weights_dict[k] = float(np.clip(class_weights_dict[k], 0.5, 3.0))
print(f"Class Weights (clipped): {class_weights_dict}")

# ------------------------------
# Model definition (EfficientNetB0 + stronger head with milder regularization)
# ------------------------------
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

# Label smoothing helps with imbalance and overconfidence
loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

# ------------------------------
# Phase 1: Train classification head only
# ------------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=loss_fn,
    metrics=['accuracy']
)
model.summary()

early_1 = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
ckpt_1 = ModelCheckpoint(str(MODELS_DIR / 'best_model.keras'), monitor='val_loss', save_best_only=True)
rlr_1 = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

history = model.fit(
    train_generator,
    epochs=EPOCHS_INITIAL,
    validation_data=val_generator,
    callbacks=[early_1, ckpt_1, rlr_1],
    class_weight=class_weights_dict,
)

# ------------------------------
# Phase 2: Fine-tune the top layers of the base model with a lower LR
# ------------------------------
base_model.trainable = True
# Unfreeze only the last 30 layers for stability
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=loss_fn,
    metrics=['accuracy']
)

early_2 = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
ckpt_2 = ModelCheckpoint(str(MODELS_DIR / 'best_model.keras'), monitor='val_loss', save_best_only=True)
rlr_2 = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

history_fine = model.fit(
    train_generator,
    epochs=EPOCHS_FINE,
    validation_data=val_generator,
    callbacks=[early_2, ckpt_2, rlr_2],
    class_weight=class_weights_dict,
)

# Save final model in modern Keras format
model.save(str(MODELS_DIR / 'final_model.keras'))

# ------------------------------
# Visualization of training curves
# ------------------------------
try:
    train_acc = history.history.get('accuracy', []) + history_fine.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', []) + history_fine.history.get('val_accuracy', [])
    train_loss = history.history.get('loss', []) + history_fine.history.get('loss', [])
    val_loss = history.history.get('val_loss', []) + history_fine.history.get('val_loss', [])

    plt.figure()
    plt.plot(train_acc, label='Train Acc')
    plt.plot(val_acc, label='Val Acc')
    plt.legend()
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.legend()
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Plotting skipped due to error: {e}")
