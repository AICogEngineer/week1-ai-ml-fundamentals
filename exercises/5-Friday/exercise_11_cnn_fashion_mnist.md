# Exercise 11: CNN for Fashion-MNIST

## Learning Objectives

- Build a complete CNN architecture
- Apply data augmentation
- Compare CNN vs Dense network performance
- Experiment with architecture variations

## Duration

**Estimated Time:** 90 minutes

---

## The Challenge

Build a CNN that classifies Fashion-MNIST images (clothing items) with **> 90% test accuracy**.

Fashion-MNIST is like MNIST but harder - 10 categories of clothing instead of digits.

**Classes:**
0. T-shirt/top
1. Trouser
2. Pullover
3. Dress
4. Coat
5. Sandal
6. Shirt
7. Sneaker
8. Bag
9. Ankle boot

---

## Part 1: Data Loading (10 min)

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Load Fashion-MNIST
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Class names for visualization
class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# TODO: Explore the data
print(f"Training: {X_train.shape}")
print(f"Test: {X_test.shape}")

# TODO: Visualize samples from each class
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, ax in enumerate(axes.flat):
    idx = np.where(y_train == i)[0][0]
    ax.imshow(X_train[idx], cmap='gray')
    ax.set_title(class_names[i])
    ax.axis('off')
plt.tight_layout()
plt.show()
```

---

## Part 2: Preprocessing (10 min)

```python
# Normalize pixel values
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Add channel dimension for CNN: (n, 28, 28) -> (n, 28, 28, 1)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

print(f"Training shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")
```

---

## Part 3: Build the CNN (20 min)

### Task 3.1: Design Your Architecture

Follow this pattern:
```
Conv2D -> ReLU -> MaxPool -> Conv2D -> ReLU -> MaxPool -> Flatten -> Dense -> Output
```

```python
model = keras.Sequential([
    # Block 1: Convolution + Pooling
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    
    # Block 2: Convolution + Pooling
    # TODO: Add second conv block
    
    # Flatten and classify
    layers.Flatten(),
    # TODO: Add dense layers
    
    # Output layer
    layers.Dense(10, activation='softmax')
], name='fashion_cnn')

model.summary()
```

### Task 3.2: Compile

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

---

## Part 4: Train (15 min)

```python
# Train with validation split
history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)
```

---

## Part 5: Evaluate (15 min)

### Task 5.1: Test Performance

```python
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc:.2%}")

if test_acc >= 0.90:
    print("SUCCESS! Target accuracy achieved!")
```

### Task 5.2: Per-Class Accuracy

```python
from sklearn.metrics import classification_report, confusion_matrix

y_pred = np.argmax(model.predict(X_test), axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))
```

### Task 5.3: Confusion Matrix

```python
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()
```

Which classes are most confused with each other?

---

## Part 6: Architecture Experiments (20 min)

Try at least 2 variations and record results:

### Variation A: More Filters

```python
model_a = keras.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])
# Train and evaluate...
```

### Variation B: Deeper Network

```python
model_b = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```

### Variation C: With Dropout

```python
model_c = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])
```

### Record Results

| Variation | Test Accuracy | Parameters | Notes |
|-----------|---------------|------------|-------|
| Baseline  |               |            |       |
| A         |               |            |       |
| B         |               |            |       |
| C         |               |            |       |

---

## Bonus: Compare to Dense Network

```python
# Build Dense-only network for comparison
model_dense = keras.Sequential([
    layers.Flatten(input_shape=(28, 28, 1)),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model_dense.compile(optimizer='adam', 
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

history_dense = model_dense.fit(X_train, y_train, epochs=15, 
                                validation_split=0.1, verbose=0)

dense_acc = model_dense.evaluate(X_test, y_test, verbose=0)[1]
print(f"Dense Network Test Accuracy: {dense_acc:.2%}")
print(f"CNN Test Accuracy: {test_acc:.2%}")
print(f"CNN improvement: {(test_acc - dense_acc):.2%}")
```

---

## Definition of Done

- [ ] Fashion-MNIST loaded and visualized
- [ ] CNN model built and compiled
- [ ] Model trained for at least 15 epochs
- [ ] Test accuracy >= 90%
- [ ] Per-class accuracy analyzed
- [ ] Confusion matrix created
- [ ] At least 2 architecture variations tested
- [ ] Comparison with Dense network completed

