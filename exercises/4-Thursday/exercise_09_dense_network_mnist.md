# Exercise 09: Dense Network for MNIST

## Learning Objectives

- Build a complete ML pipeline from scratch
- Load and preprocess image data
- Construct and train a Dense neural network
- Evaluate and interpret results

## Duration

**Estimated Time:** 90 minutes

---

## The Challenge

Build a neural network that recognizes handwritten digits (0-9) using only Dense layers (no convolutions yet - that's Friday!).

**Target accuracy: > 95% on test set**

---

## Part 1: Data Loading and Exploration (15 min)

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# TODO: Explore the data
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
print(f"Label range: {y_train.min()} to {y_train.max()}")
print(f"Pixel value range: {X_train.min()} to {X_train.max()}")

# TODO: Visualize some samples
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i], cmap='gray')
    ax.set_title(f"Label: {y_train[i]}")
    ax.axis('off')
plt.tight_layout()
plt.show()
```

---

## Part 2: Data Preprocessing (15 min)

### Task 2.1: Normalize Pixel Values

```python
# TODO: Normalize to [0, 1] range
# Why? Neural networks work better with small values
X_train_norm = None  # Divide by 255.0
X_test_norm = None
```

### Task 2.2: Flatten Images

Dense layers expect 1D input, but images are 2D (28x28).

```python
# TODO: Reshape from (n, 28, 28) to (n, 784)
X_train_flat = None
X_test_flat = None

print(f"Flattened shape: {X_train_flat.shape}")
```

### Task 2.3: Verify Preprocessing

```python
# Sanity checks
assert X_train_flat.shape == (60000, 784)
assert X_test_flat.shape == (10000, 784)
assert X_train_flat.max() <= 1.0
assert X_train_flat.min() >= 0.0
print("Preprocessing checks passed!")
```

---

## Part 3: Build the Model (20 min)

### Task 3.1: Design Your Architecture

Consider:
- Input layer: 784 features
- Hidden layers: How many? How many neurons?
- Output layer: 10 classes (digits 0-9)
- Activations: ReLU for hidden, Softmax for output

```python
model = keras.Sequential([
    # TODO: Design your architecture
    # Start simple, then add complexity if needed
], name='mnist_dense')

model.summary()
```

### Task 3.2: Calculate Expected Parameters

Before running summary(), estimate:
- If Layer 1 has 128 neurons: 784 * 128 + 128 = ?
- If Layer 2 has 64 neurons: 128 * 64 + 64 = ?
- Output layer: 64 * 10 + 10 = ?

Compare your estimate to `model.summary()`!

### Task 3.3: Compile

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

Questions:
- Why `sparse_categorical_crossentropy` instead of `categorical_crossentropy`?
- What does 'adam' optimizer do?

---

## Part 4: Train the Model (20 min)

### Task 4.1: Initial Training

```python
history = model.fit(
    X_train_flat, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)
```

### Task 4.2: Plot Training History

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss curve
axes[0].plot(history.history['loss'], label='Training')
axes[0].plot(history.history['val_loss'], label='Validation')
axes[0].set_title('Loss Over Time')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()

# Accuracy curve
axes[1].plot(history.history['accuracy'], label='Training')
axes[1].plot(history.history['val_accuracy'], label='Validation')
axes[1].set_title('Accuracy Over Time')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()

plt.tight_layout()
plt.show()
```

### Task 4.3: Analyze the Curves

Answer:
- Is the model overfitting? How can you tell?
- Did validation accuracy plateau?
- Should you train for more epochs?

---

## Part 5: Evaluate (10 min)

### Task 5.1: Test Set Performance

```python
test_loss, test_acc = model.evaluate(X_test_flat, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.2%}")

# Did you meet the target?
if test_acc >= 0.95:
    print("SUCCESS! Target accuracy achieved!")
else:
    print(f"Keep trying! Need {(0.95 - test_acc):.2%} more accuracy")
```

### Task 5.2: Confusion Analysis

```python
# Get predictions
y_pred = model.predict(X_test_flat)
y_pred_classes = np.argmax(y_pred, axis=1)

# Find misclassified examples
misclassified = np.where(y_pred_classes != y_test)[0]
print(f"Misclassified: {len(misclassified)} / {len(y_test)}")

# TODO: Visualize some misclassified examples
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, idx in enumerate(misclassified[:10]):
    ax = axes[i // 5, i % 5]
    ax.imshow(X_test[idx], cmap='gray')
    ax.set_title(f"True: {y_test[idx]}, Pred: {y_pred_classes[idx]}")
    ax.axis('off')
plt.suptitle("Misclassified Examples")
plt.tight_layout()
plt.show()
```

---

## Part 6: Improve the Model (10 min)

If you didn't reach 95%, try these improvements:

### Option A: Add More Layers

```python
# model_v2 = keras.Sequential([
#     layers.Dense(256, activation='relu', input_shape=(784,)),
#     layers.Dense(128, activation='relu'),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(10, activation='softmax')
# ])
```

### Option B: Add Dropout

```python
# model_v2 = keras.Sequential([
#     layers.Dense(256, activation='relu', input_shape=(784,)),
#     layers.Dropout(0.3),
#     layers.Dense(128, activation='relu'),
#     layers.Dropout(0.3),
#     layers.Dense(10, activation='softmax')
# ])
```

### Option C: Add Batch Normalization

```python
# model_v2 = keras.Sequential([
#     layers.Dense(256, input_shape=(784,)),
#     layers.BatchNormalization(),
#     layers.Activation('relu'),
#     layers.Dense(128),
#     layers.BatchNormalization(),
#     layers.Activation('relu'),
#     layers.Dense(10, activation='softmax')
# ])
```

---

## Bonus Challenges

### Challenge A: Learning Rate Comparison

Train with different learning rates and compare:
```python
# lr_values = [0.001, 0.01, 0.1]
# for lr in lr_values:
#     model = create_model()
#     model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), ...)
```

### Challenge B: Architecture Search

Try at least 3 different architectures and compare test accuracy.

### Challenge C: Save and Load Model

```python
# Save model
model.save('mnist_dense.keras')

# Load model
loaded_model = keras.models.load_model('mnist_dense.keras')
```

---

## Definition of Done

- [ ] Data loaded and preprocessed correctly
- [ ] Model built with Dense layers only
- [ ] Model trained for at least 10 epochs
- [ ] Training curves plotted and analyzed
- [ ] Test accuracy achieved >= 95%
- [ ] Misclassified examples visualized
- [ ] At least one improvement attempted

---

## Expected Results

A well-designed Dense network should achieve:
- Training accuracy: ~98-99%
- Validation accuracy: ~97-98%
- Test accuracy: ~97%

Note: CNNs (Friday's topic) can achieve >99%!

