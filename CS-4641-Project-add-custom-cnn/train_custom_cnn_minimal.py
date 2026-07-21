import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    cohen_kappa_score, balanced_accuracy_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

print(f"Using TensorFlow Version: {tf.__version__}")

# Config
ROOT = '/Users/adilhusain/ML-Project'
DATA_DIR = 'data/'
TRAIN_IMG_DIR = os.path.join(DATA_DIR, 'train_images')
CSV_PATH = os.path.join(DATA_DIR, 'train.csv')

IMG_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 25
NUM_CLASSES = 5
LR = 1e-3

RUN_DIR = os.path.join(ROOT, 'runs_cnn')
OUT_DIR = os.path.join(RUN_DIR, 'baseline_custom')
os.makedirs(OUT_DIR, exist_ok=True)

# standardization + augmentation
print("Loading and preparing data...")
df = pd.read_csv(CSV_PATH)
df['id_code'] = df['id_code'].astype(str) + '.png'
df['diagnosis'] = df['diagnosis'].astype(int).astype(str)

train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=SEED, stratify=df['diagnosis']
)

print(f"Total training images: {len(train_df)}")
print(f"Total validation images: {len(val_df)}")

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255.,
    rotation_range=30,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.1,
    brightness_range=[0.8, 1.2]
)

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.)

print("Creating data generators...")
train_gen = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=TRAIN_IMG_DIR,
    x_col='id_code',
    y_col='diagnosis',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    seed=SEED
)

val_gen = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=TRAIN_IMG_DIR,
    x_col='id_code',
    y_col='diagnosis',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

class_indices = train_gen.class_indices
idx_to_class = {v: k for k, v in class_indices.items()}

# class weights
train_counts = train_df['diagnosis'].value_counts().sort_index()
max_count = train_counts.max()
class_weight = {i: float(max_count / c) for i, c in enumerate(train_counts.tolist())}
print("Class counts:", train_counts.to_dict())
print("Class weight:", class_weight)

# model
print("Building custom CNN model...")
model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    # block 1
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),

    # block 2
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),

    # block 3
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.30),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

# callbacks
ckpt_path = os.path.join(OUT_DIR, 'best_model.h5')
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True, verbose=1),
    tf.keras.callbacks.CSVLogger(os.path.join(OUT_DIR, 'training_log.csv'))
]

# training
print("Starting model training...")
history = model.fit(
    train_gen,
    steps_per_epoch=train_gen.samples // BATCH_SIZE,
    validation_data=val_gen,
    validation_steps=val_gen.samples // BATCH_SIZE,
    epochs=EPOCHS,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1
)

print("Training complete.")
final_path = os.path.join(OUT_DIR, 'final_model.h5')
model.save(final_path)
print(f"Saved final model to {final_path}")

# metrics
print("\n--- Model Evaluation ---")
y_true = val_gen.classes
y_prob = model.predict(val_gen, verbose=0)
y_pred = np.argmax(y_prob, axis=1)
class_labels = [idx_to_class[i] for i in range(NUM_CLASSES)]

kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')
balanced_acc = balanced_accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=class_labels, digits=4)

print(f"Cohen's Kappa (Quadratic): {kappa:.4f}")
print(f"Balanced Accuracy: {balanced_acc:.4f}")
print("\nClassification Report:\n", report)

with open(os.path.join(OUT_DIR, 'metrics.json'), 'w') as f:
    json.dump({
        "kappa_quadratic": float(kappa),
        "balanced_accuracy": float(balanced_acc),
        "val_accuracy": [float(x) for x in history.history.get('val_accuracy', [])],
        "val_loss": [float(x) for x in history.history.get('val_loss', [])]
    }, f, indent=2)

with open(os.path.join(OUT_DIR, 'classification_report.txt'), 'w') as f:
    f.write(report)

# visualizations
print("Generating and saving plots...")

plt.figure(figsize=(12, 5))

# accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()

# loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
plt.tight_layout()
acc_loss_path = os.path.join(OUT_DIR, 'baseline_history.png')
plt.savefig(acc_loss_path, dpi=200)
print(f"Saved training history plot to '{acc_loss_path}'")

# confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation='horizontal', values_format='d')
plt.title('Baseline CNN Confusion Matrix')
cm_path = os.path.join(OUT_DIR, 'baseline_confusion_matrix.png')
plt.savefig(cm_path, dpi=200, bbox_inches='tight')
print(f"Saved confusion matrix to '{cm_path}'")

print("\nScript finished. Artifacts in:", OUT_DIR)