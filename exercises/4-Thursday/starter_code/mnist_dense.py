"""
Exercise 09: Dense Network for MNIST
====================================

Build and train a Dense neural network for digit classification.
Target: > 95% test accuracy
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("MNIST DIGIT CLASSIFICATION WITH DENSE NETWORK")
print("=" * 60)

# =============================================================================
# PART 1: DATA LOADING AND EXPLORATION
# =============================================================================

print("\n--- LOADING DATA ---")

# Load MNIST
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# TODO: Print data shapes and ranges
# print(f"Training set: {X_train.shape}")
# print(f"Test set: {X_test.shape}")
# print(f"Labels: {y_train.min()} to {y_train.max()}")
# print(f"Pixel range: {X_train.min()} to {X_train.max()}")

# TODO: Visualize samples
# fig, axes = plt.subplots(2, 5, figsize=(12, 5))
# for i, ax in enumerate(axes.flat):
#     ax.imshow(X_train[i], cmap='gray')
#     ax.set_title(f"Label: {y_train[i]}")
#     ax.axis('off')
# plt.tight_layout()
# plt.show()

# =============================================================================
# PART 2: DATA PREPROCESSING
# =============================================================================

print("\n--- PREPROCESSING ---")

# TODO: Normalize to [0, 1]
# X_train_norm = X_train.astype('float32') / 255.0
# X_test_norm = X_test.astype('float32') / 255.0

# TODO: Flatten images from (n, 28, 28) to (n, 784)
# X_train_flat = X_train_norm.reshape(-1, 784)
# X_test_flat = X_test_norm.reshape(-1, 784)

# print(f"Flattened training shape: {X_train_flat.shape}")

# =============================================================================
# PART 3: BUILD THE MODEL
# =============================================================================

print("\n--- BUILDING MODEL ---")

# TODO: Design your architecture
model = keras.Sequential([
    # layers.Dense(???, activation='relu', input_shape=(784,)),
    # layers.Dense(???, activation='relu'),
    # layers.Dense(10, activation='softmax')
], name='mnist_dense')

# model.summary()

# TODO: Compile
# model.compile(
#     optimizer='adam',
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy']
# )

# =============================================================================
# PART 4: TRAIN THE MODEL
# =============================================================================

print("\n--- TRAINING ---")

# TODO: Train
# history = model.fit(
#     X_train_flat, y_train,
#     epochs=10,
#     batch_size=32,
#     validation_split=0.1,
#     verbose=1
# )

# TODO: Plot training history
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# 
# axes[0].plot(history.history['loss'], label='Training')
# axes[0].plot(history.history['val_loss'], label='Validation')
# axes[0].set_title('Loss')
# axes[0].legend()
# 
# axes[1].plot(history.history['accuracy'], label='Training')
# axes[1].plot(history.history['val_accuracy'], label='Validation')
# axes[1].set_title('Accuracy')
# axes[1].legend()
# 
# plt.tight_layout()
# plt.show()

# =============================================================================
# PART 5: EVALUATE
# =============================================================================

print("\n--- EVALUATION ---")

# TODO: Evaluate on test set
# test_loss, test_acc = model.evaluate(X_test_flat, y_test, verbose=0)
# print(f"Test Accuracy: {test_acc:.2%}")
# 
# if test_acc >= 0.95:
#     print("SUCCESS! Target achieved!")
# else:
#     print(f"Need {(0.95 - test_acc):.2%} more accuracy")

# TODO: Find and visualize misclassified examples
# y_pred = model.predict(X_test_flat)
# y_pred_classes = np.argmax(y_pred, axis=1)
# misclassified = np.where(y_pred_classes != y_test)[0]
# print(f"Misclassified: {len(misclassified)}")

# =============================================================================
# PART 6: IMPROVEMENTS
# =============================================================================

print("\n--- IMPROVEMENTS ---")

# Try different architectures if you haven't reached 95%:
# 
# Option A: More layers
# Option B: Dropout
# Option C: Batch Normalization
#
# Record your best result:
# Best architecture: 
# Best test accuracy: 

