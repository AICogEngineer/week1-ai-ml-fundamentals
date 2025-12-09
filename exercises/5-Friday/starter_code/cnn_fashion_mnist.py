"""
Exercise 11: CNN for Fashion-MNIST
==================================

Build and train a CNN for clothing classification.
Target: > 90% test accuracy
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("CNN FOR FASHION-MNIST")
print("=" * 60)

# =============================================================================
# PART 1: LOAD DATA
# =============================================================================

print("\n--- LOADING DATA ---")

(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(f"Training: {X_train.shape}")
print(f"Test: {X_test.shape}")

# TODO: Visualize samples
# fig, axes = plt.subplots(2, 5, figsize=(15, 6))
# for i, ax in enumerate(axes.flat):
#     idx = np.where(y_train == i)[0][0]
#     ax.imshow(X_train[idx], cmap='gray')
#     ax.set_title(class_names[i])
#     ax.axis('off')
# plt.tight_layout()
# plt.show()

# =============================================================================
# PART 2: PREPROCESS
# =============================================================================

print("\n--- PREPROCESSING ---")

# Normalize
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Add channel dimension
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

print(f"Training shape: {X_train.shape}")

# =============================================================================
# PART 3: BUILD CNN
# =============================================================================

print("\n--- BUILDING CNN ---")

model = keras.Sequential([
    # TODO: Add Conv + Pool blocks
    # layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    # layers.MaxPooling2D((2, 2)),
    # ...
    
    # TODO: Add Flatten and Dense layers
    # layers.Flatten(),
    # layers.Dense(64, activation='relu'),
    # layers.Dense(10, activation='softmax')
], name='fashion_cnn')

# model.summary()

# model.compile(
#     optimizer='adam',
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy']
# )

# =============================================================================
# PART 4: TRAIN
# =============================================================================

print("\n--- TRAINING ---")

# history = model.fit(
#     X_train, y_train,
#     epochs=15,
#     batch_size=64,
#     validation_split=0.1,
#     verbose=1
# )

# =============================================================================
# PART 5: EVALUATE
# =============================================================================

print("\n--- EVALUATION ---")

# test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
# print(f"Test Accuracy: {test_acc:.2%}")

# TODO: Classification report
# TODO: Confusion matrix

# =============================================================================
# PART 6: ARCHITECTURE EXPERIMENTS
# =============================================================================

print("\n--- ARCHITECTURE EXPERIMENTS ---")

# Record your results:
# | Variation | Test Accuracy | Parameters |
# |-----------|---------------|------------|
# | Baseline  |               |            |
# | A         |               |            |
# | B         |               |            |

# =============================================================================
# BONUS: COMPARE TO DENSE
# =============================================================================

# model_dense = keras.Sequential([
#     layers.Flatten(input_shape=(28, 28, 1)),
#     layers.Dense(256, activation='relu'),
#     layers.Dense(128, activation='relu'),
#     layers.Dense(10, activation='softmax')
# ])
#
# Compare accuracies...

