import os
import json
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.efficientnet import preprocess_input
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------
# Paths and constants
# ------------------------------
TEST_DIR = '../data/test'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
MODELS_DIR = '../models'
MODEL_CANDIDATES = [
    os.path.join(MODELS_DIR, 'best_model.keras'),  # preferred new format
    os.path.join(MODELS_DIR, 'best_model.h5'),     # legacy fallback
]
CLASS_INDICES_PATH = os.path.join(MODELS_DIR, 'class_indices.json')

# ------------------------------
# Load model (prefer .keras)
# ------------------------------
model_path = None
for candidate in MODEL_CANDIDATES:
    if os.path.exists(candidate):
        model_path = candidate
        break

if model_path is None:
    raise FileNotFoundError(
        f"No model file found. Looked for: {MODEL_CANDIDATES}. Train the model first."
    )

print(f"Loading model from: {model_path}")
model = tf.keras.models.load_model(model_path)

# ------------------------------
# Data pipeline (EfficientNet preprocessing)
# ------------------------------
if not os.path.exists(TEST_DIR):
    raise FileNotFoundError(f"Expected test directory at {TEST_DIR}")

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
)

# ------------------------------
# Load training class indices and verify mapping
# ------------------------------
train_class_indices = None
if os.path.exists(CLASS_INDICES_PATH):
    with open(CLASS_INDICES_PATH) as f:
        train_class_indices = json.load(f)
    print("Loaded training class mapping:", train_class_indices)
else:
    print(f"Warning: {CLASS_INDICES_PATH} not found. Proceeding with test mapping only.")

test_class_indices = test_generator.class_indices
print("Test class mapping:", test_class_indices)

# Build class label order and true label remapping to align with training
if train_class_indices is not None:
    # Verify same class names present
    if set(train_class_indices.keys()) != set(test_class_indices.keys()):
        raise RuntimeError(
            "Class names differ between train and test directories. "
            f"Train: {sorted(train_class_indices.keys())}, Test: {sorted(test_class_indices.keys())}"
        )

    # Map test-index -> train-index so true labels align with model's training order
    remap = np.zeros(len(train_class_indices), dtype=int)
    for cls_name, train_idx in train_class_indices.items():
        test_idx = test_class_indices[cls_name]
        remap[test_idx] = train_idx

    # True labels as provided by test generator (in test index space)
    true_classes_raw = test_generator.classes
    true_classes = remap[true_classes_raw]

    # Class labels in training index order
    inv_train = {v: k for k, v in train_class_indices.items()}
    class_labels = [inv_train[i] for i in range(len(inv_train))]
else:
    # Fallback: assume mapping equals test mapping (may be wrong if different order was used in training)
    true_classes = test_generator.classes
    inv_test = {v: k for k, v in test_class_indices.items()}
    class_labels = [inv_test[i] for i in range(len(inv_test))]

# ------------------------------
# Predict and compute metrics aligned with training mapping
# ------------------------------
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)

# Compute accuracy with aligned labels
acc = accuracy_score(true_classes, predicted_classes)
print(f"Test Accuracy (aligned): {acc:.4f}")

# Confusion matrix in training index order
labels_range = list(range(len(class_labels)))
cm = confusion_matrix(true_classes, predicted_classes, labels=labels_range)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Aligned)')
plt.tight_layout()
plt.show()

# Detailed classification report (silence zero-division warnings)
report = classification_report(true_classes, predicted_classes, labels=labels_range, target_names=class_labels, zero_division=0)
print(report)
