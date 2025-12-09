# Keras Sequential API

## Learning Objectives

- Understand Keras as TensorFlow's high-level neural network API
- Build models using the Sequential API by stacking layers
- Interpret model summaries and understand parameter counts
- Compile models with appropriate optimizers, loss functions, and metrics

## Why This Matters

You've learned how neural networks work mathematically - forward propagation, activation functions, loss computation. Now it's time to stop implementing everything from scratch and use professional tools.

Keras is TensorFlow's answer to the question: "How do we build neural networks without writing hundreds of lines of boilerplate?" With Keras, you can define the same networks you'd build manually, but in just a few lines of clear, readable code.

In our **From Zero to Neural** journey, Keras is where theory becomes practical. The concepts you've learned this week directly map to Keras components.

## The Concept

### What Is Keras?

**Keras** is a high-level neural network API that:
- Is built into TensorFlow 2.x (no separate installation needed)
- Provides intuitive abstractions for building models
- Handles boilerplate (gradients, updates, batching)
- Supports experimentation with minimal code changes

```
Abstraction Hierarchy:

TensorFlow Core (Low-level)
    |
    v
tf.keras (High-level)
    |
    +-- Sequential API  <-- Simplest, linear stack of layers
    +-- Functional API  <-- Flexible, multiple inputs/outputs
    +-- Model Subclassing  <-- Full control, custom forward pass
```

### The Sequential Model

The **Sequential** model is perfect for networks where data flows straight through layers, one after another:

```
Input -> Layer 1 -> Layer 2 -> Layer 3 -> Output

No branches, no skips, no multiple inputs/outputs.
```

**Creating a Sequential Model:**

```python
from tensorflow import keras
from tensorflow.keras import layers

# Method 1: Pass layers as list
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(784,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Method 2: Add layers incrementally
model = keras.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(784,)))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
```

### Anatomy of a Sequential Model

```python
# MNIST digit classification network

model = keras.Sequential([
    # Layer 1: 784 inputs -> 128 neurons, ReLU activation
    layers.Dense(128, activation='relu', input_shape=(784,)),
    
    # Layer 2: 128 -> 64 neurons, ReLU activation
    layers.Dense(64, activation='relu'),
    
    # Output: 64 -> 10 classes, Softmax for probabilities
    layers.Dense(10, activation='softmax')
])
```

**Mapping to Concepts You Learned:**

| Keras Code | Neural Network Concept |
|------------|----------------------|
| `Dense(128)` | Layer with 128 neurons |
| `activation='relu'` | ReLU activation function |
| `input_shape=(784,)` | Input layer size |
| `Dense(10, activation='softmax')` | Output layer with softmax |

### Understanding Model Summary

```python
model.summary()
```

**Output:**
```
Model: "sequential"
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
dense (Dense)               (None, 128)               100480    
dense_1 (Dense)             (None, 64)                8256      
dense_2 (Dense)             (None, 10)                650       
=================================================================
Total params: 109,386
Trainable params: 109,386
Non-trainable params: 0
_________________________________________________________________
```

**Interpreting the Summary:**

| Column | Meaning |
|--------|---------|
| **Layer (type)** | Layer name and class |
| **Output Shape** | (batch_size, output_features) - None means any batch size |
| **Param #** | Number of trainable parameters |

**Parameter Count Formula:**
```
Parameters = (input_features * neurons) + neurons
             (weights)                    (biases)

Layer 1: (784 * 128) + 128 = 100,480
Layer 2: (128 * 64) + 64 = 8,256
Layer 3: (64 * 10) + 10 = 650
Total: 109,386
```

### Compiling the Model

Before training, you must **compile** the model with:
1. **Optimizer**: How to update weights
2. **Loss function**: What to minimize
3. **Metrics**: What to track during training

```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

**Common Optimizers:**

| Optimizer | Use Case |
|-----------|----------|
| `'sgd'` | Simple, requires tuning learning rate |
| `'adam'` | Default choice, adaptive learning rate |
| `'rmsprop'` | Good for RNNs |

**Common Loss Functions:**

| Loss | Use Case |
|------|----------|
| `'binary_crossentropy'` | Binary classification |
| `'categorical_crossentropy'` | Multi-class (one-hot labels) |
| `'sparse_categorical_crossentropy'` | Multi-class (integer labels) |
| `'mse'` | Regression |

**Common Metrics:**

| Metric | Meaning |
|--------|---------|
| `'accuracy'` | Classification accuracy |
| `'mae'` | Mean Absolute Error (regression) |
| `'precision'`, `'recall'` | Binary classification |

### Training with fit()

```python
history = model.fit(
    X_train,           # Training features
    y_train,           # Training labels
    epochs=10,         # Number of passes through data
    batch_size=32,     # Samples per gradient update
    validation_data=(X_val, y_val),  # Optional validation set
    verbose=1          # Progress output level
)
```

**Training Output:**
```
Epoch 1/10
1875/1875 [==============================] - 3s 2ms/step - loss: 0.2543 - accuracy: 0.9265 - val_loss: 0.1234 - val_accuracy: 0.9612
Epoch 2/10
1875/1875 [==============================] - 3s 2ms/step - loss: 0.1045 - accuracy: 0.9687 - val_loss: 0.0987 - val_accuracy: 0.9701
...
```

**History Object:**
```python
# Access training history
print(history.history.keys())
# dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])

# Plot training curves
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()
```

### Making Predictions

```python
# Predict probabilities
probabilities = model.predict(X_test)

# Get class predictions
predictions = probabilities.argmax(axis=1)

# For single sample
single_prediction = model.predict(X_test[:1])
```

### Evaluating the Model

```python
# Evaluate on test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.2%}")
```

## Code Example: Complete Sequential Model

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

print("=" * 60)
print("KERAS SEQUENTIAL API")
print("=" * 60)

# === Generate Sample Data ===
print("\n--- Creating Sample Data ---")

# Simulating MNIST-like data
n_samples = 1000
n_features = 784  # 28x28 flattened
n_classes = 10

X = np.random.randn(n_samples, n_features).astype(np.float32)
y = np.random.randint(0, n_classes, n_samples)

# Split into train/test
split = int(0.8 * n_samples)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print(f"Features per sample: {X_train.shape[1]}")
print(f"Number of classes: {n_classes}")

# === Build Sequential Model ===
print("\n--- Building Model ---")

model = keras.Sequential([
    # Input layer is implicit; first Dense specifies input_shape
    layers.Dense(128, activation='relu', input_shape=(n_features,), name='hidden_1'),
    layers.Dense(64, activation='relu', name='hidden_2'),
    layers.Dense(32, activation='relu', name='hidden_3'),
    layers.Dense(n_classes, activation='softmax', name='output')
], name='digit_classifier')

print("Model created!")

# === Model Summary ===
print("\n--- Model Summary ---")
model.summary()

# === Compile Model ===
print("\n--- Compiling Model ---")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',  # Integer labels
    metrics=['accuracy']
)

print("Optimizer: Adam (lr=0.001)")
print("Loss: Sparse Categorical Crossentropy")
print("Metrics: Accuracy")

# === Train Model ===
print("\n--- Training Model ---")

history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2,  # Use 20% of training data for validation
    verbose=1
)

# === Evaluate Model ===
print("\n--- Evaluation ---")

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2%}")

# === Make Predictions ===
print("\n--- Predictions ---")

# Predict on first 5 test samples
predictions = model.predict(X_test[:5], verbose=0)
predicted_classes = predictions.argmax(axis=1)
actual_classes = y_test[:5]

print("Sample predictions:")
for i, (pred, actual, probs) in enumerate(zip(predicted_classes, actual_classes, predictions)):
    confidence = probs.max() * 100
    status = "Correct" if pred == actual else "Wrong"
    print(f"  Sample {i+1}: Predicted={pred}, Actual={actual}, Confidence={confidence:.1f}% - {status}")

# === Access Training History ===
print("\n--- Training History ---")
print(f"Final training loss: {history.history['loss'][-1]:.4f}")
print(f"Final training accuracy: {history.history['accuracy'][-1]:.2%}")
print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.2%}")

# === Layer Access ===
print("\n--- Layer Information ---")
for i, layer in enumerate(model.layers):
    weights = layer.get_weights()
    if weights:
        W, b = weights[0], weights[1]
        print(f"Layer {i} ({layer.name}):")
        print(f"  Weight shape: {W.shape}")
        print(f"  Bias shape: {b.shape}")
        print(f"  Total params: {W.size + b.size}")

print("\n" + "=" * 60)
```

**Sample Output:**
```
============================================================
KERAS SEQUENTIAL API
============================================================

--- Creating Sample Data ---
Training samples: 800
Test samples: 200
Features per sample: 784
Number of classes: 10

--- Building Model ---
Model created!

--- Model Summary ---
Model: "digit_classifier"
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
hidden_1 (Dense)            (None, 128)               100480    
hidden_2 (Dense)            (None, 64)                8256      
hidden_3 (Dense)            (None, 32)                2080      
output (Dense)              (None, 10)                330       
=================================================================
Total params: 111,146
Trainable params: 111,146
Non-trainable params: 0
_________________________________________________________________

--- Compiling Model ---
Optimizer: Adam (lr=0.001)
Loss: Sparse Categorical Crossentropy
Metrics: Accuracy

--- Training Model ---
Epoch 1/5
20/20 [==============================] - 1s 15ms/step - loss: 2.4521 - accuracy: 0.0844 - val_loss: 2.3421 - val_accuracy: 0.0938
...

--- Evaluation ---
Test Loss: 2.3012
Test Accuracy: 11.50%

--- Predictions ---
Sample predictions:
  Sample 1: Predicted=7, Actual=3, Confidence=15.2% - Wrong
  ...

============================================================
```

*Note: Random data won't learn meaningful patterns - this demonstrates the API, not achieving high accuracy.*

### Sequential Model Patterns

**Binary Classification:**
```python
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(features,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Single output, sigmoid
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

**Regression:**
```python
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(features,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # No activation for regression output
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
```

## Key Takeaways

1. **Sequential is for linear layer stacks** - when data flows straight through without branches.

2. **Dense layers are fully connected** - every input connects to every neuron.

3. **compile() sets training configuration** - optimizer, loss, and metrics.

4. **fit() trains the model** - specify epochs, batch size, and validation data.

5. **Model summary shows architecture** - output shapes and parameter counts.

## Looking Ahead

The next reading dives deep into **Dense layers** - the most fundamental layer type. You'll understand exactly what happens inside a Dense layer and how it maps to the MLP concepts from Wednesday.

## Additional Resources

- [Keras Sequential Guide](https://www.tensorflow.org/guide/keras/sequential_model) - Official documentation
- [Keras Layer Catalog](https://keras.io/api/layers/) - All available layers
- [Keras Model Training](https://www.tensorflow.org/guide/keras/train_and_evaluate) - fit(), evaluate(), predict()

