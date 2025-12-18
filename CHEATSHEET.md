# Weekly Cheatsheet: AI-ML-Fundamentals


## Weekly Overview

This week transformed trainees from AI consumers into AI practitioners. We covered machine learning fundamentals from regression/classification through neural networks to CNNs, building mathematical intuition at each step. By the end, trainees can:

- Distinguish supervised vs unsupervised learning and choose appropriate algorithms
- Build and train neural networks using TensorFlow/Keras Sequential API
- Design CNN architectures for image classification
- Interpret training metrics to diagnose model issues (overfitting, underfitting)

---

## Concept Quick Reference

| Concept | Definition | Key Use Case |
|---------|------------|--------------|
| Supervised Learning | Learning from labeled data (input + correct output) | Classification, regression tasks |
| Unsupervised Learning | Finding patterns in unlabeled data | Clustering, anomaly detection |
| Regression | Predicting continuous values | House prices, temperature, stock prices |
| Classification | Predicting discrete categories | Spam detection, medical diagnosis |
| K-Means Clustering | Grouping data into K clusters by distance to centroids | Customer segmentation, image compression |
| Perceptron | Single artificial neuron with weighted inputs and activation | Binary classification, building block for MLPs |
| Activation Function | Non-linear function applied after weighted sum | Enabling networks to learn complex patterns |
| Forward Propagation | Data flowing through network layers to produce output | Making predictions |
| Loss Function | Measures prediction error (how wrong the model is) | Guiding learning via gradient descent |
| Dense Layer | Fully connected layer where every input connects to every output | Classification head, feature combination |
| Conv2D | Convolutional layer that applies learnable filters to detect features | Image feature extraction |
| Pooling | Downsampling operation reducing spatial dimensions | Translation invariance, dimensionality reduction |
| Flatten | Reshaping multi-dimensional output to 1D for Dense layers | Bridge between CNN and classification head |

---

## Pros & Cons

### Supervised vs Unsupervised Learning

| Approach | Pros | Cons | Best For |
|----------|------|------|----------|
| Supervised | Precise predictions, measurable accuracy | Requires labeled data (expensive) | When you have labels and want predictions |
| Unsupervised | No labels needed, discovers hidden patterns | Harder to evaluate, subjective interpretation | Exploratory analysis, when labels unavailable |

### Activation Functions

| Function | Pros | Cons | Best For |
|----------|------|------|----------|
| ReLU | Fast, avoids vanishing gradient, sparse activation | Dead neurons (output stuck at 0) | Hidden layers (default choice) |
| Leaky ReLU | Prevents dead neurons, small gradient for negatives | Extra hyperparameter (alpha) | When ReLU neurons are dying |
| Sigmoid | Outputs probability (0-1), smooth gradient | Vanishing gradient, not zero-centered | Output layer (binary classification) |
| Tanh | Zero-centered, outputs (-1, 1) | Still has vanishing gradient | RNNs, when need centered outputs |
| Softmax | Outputs probability distribution (sums to 1) | Only for output layer | Multi-class classification output |

### ML Algorithm Comparison

| Algorithm | Pros | Cons | Best For |
|-----------|------|------|----------|
| Linear Regression | Fast, interpretable, simple | Only linear relationships | Continuous prediction with linear patterns |
| Logistic Regression | Fast, interpretable, outputs probabilities | Linear decision boundary only | Binary classification, baseline model |
| Decision Tree | No preprocessing needed, interpretable | Prone to overfitting | Explainable models, mixed feature types |
| K-Means | Simple, fast, scalable | Must specify K, sensitive to initialization | Customer segmentation, data compression |
| Neural Network | Learns complex patterns, universal approximator | Black box, requires tuning and data | Complex patterns, image/text/audio |
| CNN | Preserves spatial relationships, translation invariant | Needs lots of data, computationally expensive | Image classification, object detection |

---

## When to Use What

### Choosing a Learning Paradigm

| If you have... | And you want to... | Then use... | Because... |
|----------------|-------------------|-------------|------------|
| Labeled data | Predict continuous values | Regression | Outputs numeric predictions |
| Labeled data | Predict categories | Classification | Outputs discrete labels |
| Unlabeled data | Find natural groupings | Clustering (K-Means) | Discovers patterns without labels |
| Image data | Classify images | CNN | Preserves spatial relationships |
| Sequential data | Capture temporal patterns | RNN/LSTM (Week 2) | Processes sequences |

### Choosing an Activation Function

| If your layer is... | And you need... | Then use... | Because... |
|---------------------|-----------------|-------------|------------|
| Hidden layer | Default choice | ReLU | Fast, effective, avoids vanishing gradient |
| Hidden layer | Prevent dead neurons | Leaky ReLU | Small gradient for negatives |
| Output (binary) | Probability 0-1 | Sigmoid | Maps to probability range |
| Output (multi-class) | Probability distribution | Softmax | Outputs sum to 1 |
| Output (regression) | Any real number | Linear (none) | No activation needed |

### Choosing CNN Parameters

| If you want... | Then set... | Because... |
|----------------|-------------|------------|
| Same output size as input | `padding='same'` | Zero-padding preserves dimensions |
| Reduced spatial size | `padding='valid'` (default) | No padding, output shrinks |
| Faster downsampling | `strides=2` | Skips positions, halves dimensions |
| More features detected | More `filters` | Each filter learns different pattern |
| Larger receptive field | Larger `kernel_size` | Sees more context per position |

---

## Essential Commands

### scikit-learn (Tuesday)

```python
# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Classification
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

# K-Means Clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_
centers = kmeans.cluster_centers_
```

### TensorFlow/Keras Sequential API (Thursday)

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Build Sequential Model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile (configure training)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # Integer labels
    metrics=['accuracy']
)

# Train
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)

# Evaluate
test_loss, test_accuracy = model.evaluate(X_test, y_test)

# Predict
predictions = model.predict(X_test)
predicted_classes = predictions.argmax(axis=1)

# View architecture
model.summary()
```

### CNN Layers (Friday)

```python
# CNN for Image Classification
model = keras.Sequential([
    # Convolutional layers
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Classification head
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Conv2D parameters
layers.Conv2D(
    filters=32,           # Number of filters (output channels)
    kernel_size=(3, 3),   # Filter size
    strides=(1, 1),       # Step size
    padding='valid',      # 'valid' (no pad) or 'same' (preserve size)
    activation='relu'
)

# Output size formula
# Output = floor((Input - Kernel + 2*Padding) / Stride) + 1
```

---

## Common Gotchas

| Topic | Wrong | Right |
|-------|-------|-------|
| Data Prep | Training on entire dataset | Always split into train/test (80/20 or 70/30) |
| Data Prep | Forgetting to normalize/scale inputs | Scale features to 0-1 or standardize (mean=0, std=1) |
| Evaluation | Using accuracy for imbalanced classes | Use F1-score, precision/recall, or balanced accuracy |
| Training | Ignoring validation loss while training | Monitor val_loss for overfitting (train ↓, val ↑ = bad) |
| Activation | Using Sigmoid in hidden layers | Use ReLU for hidden layers (faster, avoids vanishing gradient) |
| CNN | Forgetting Flatten before Dense layers | Always Flatten between Conv2D and Dense |
| Keras | Using `categorical_crossentropy` with integer labels | Use `sparse_categorical_crossentropy` for integer labels |
| Keras | Not specifying `input_shape` on first layer | First layer needs `input_shape=(features,)` or `input_shape=(H, W, C)` |
| Training | Setting learning rate too high | Start with optimizer defaults (Adam: 0.001) |
| Training | Training for too few epochs | Train until validation loss stops improving |

---

## Key Formulas

### Mean Squared Error (MSE) - Regression Loss
```
MSE = (1/n) * Σ(y_actual - y_predicted)²

Where:
- n = number of samples
- Lower is better
```

### Cross-Entropy Loss - Classification Loss
```
Binary: L = -[y*log(p) + (1-y)*log(1-p)]
Categorical: L = -Σ y_i * log(p_i)

Where:
- y = true label (0 or 1)
- p = predicted probability
- Lower is better
```

### Dense Layer Parameters
```
Parameters = (input_size × output_size) + output_size
           = (inputs × neurons) + biases

Example: Dense(64) with 128 inputs
Params = (128 × 64) + 64 = 8,256
```

### Conv2D Output Size
```
Output = floor((Input - Kernel + 2*Padding) / Stride) + 1

Examples:
- Input: 28, Kernel: 3, Pad: 0, Stride: 1 → Output: 26
- Input: 28, Kernel: 3, Pad: 1, Stride: 1 → Output: 28 (same)
- Input: 28, Kernel: 3, Pad: 0, Stride: 2 → Output: 13
```

### Conv2D Parameters
```
Parameters = (kernel_h × kernel_w × input_channels + 1) × filters

Example: Conv2D(32, (3,3)) with 1 input channel
Params = (3 × 3 × 1 + 1) × 32 = 320
```
