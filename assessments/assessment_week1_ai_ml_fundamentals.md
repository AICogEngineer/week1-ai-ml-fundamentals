# Weekly Technical Assessment: AI/ML Fundamentals

> **Week 1 Comprehensive Assessment**
> **Time Limit:** 90 minutes
> **Total Points:** 100
> **Passing Score:** 70%

---

## Section A: Conceptual Understanding (30 Points)

*Answer each question concisely. Focus on demonstrating understanding, not length.*

### A1. Machine Learning Foundations (6 points)

**A1a.** (2 pts) Define machine learning in one sentence and explain what distinguishes it from traditional programming.

<details>
<summary>Model Answer</summary>

Machine learning is a field where algorithms learn patterns from data to make predictions without being explicitly programmed. The key distinction is:
- **Traditional Programming:** Rules are written by humans (Input + Rules = Output)
- **Machine Learning:** Rules are learned from data (Input + Output = Rules/Model)
</details>

**A1b.** (2 pts) Classify each scenario as supervised or unsupervised learning. Justify briefly.
- Scenario 1: Predicting house prices from features like bedrooms and location
- Scenario 2: Grouping news articles by topic without predefined categories

<details>
<summary>Model Answer</summary>

- **Scenario 1: Supervised Learning** - We have labeled examples (historical prices) to learn from. The model learns the mapping from features to price.
- **Scenario 2: Unsupervised Learning** - No predefined labels exist. The algorithm discovers natural groupings based on article similarity/content.
</details>

**A1c.** (2 pts) Explain the difference between regression and classification with one example each.

<details>
<summary>Model Answer</summary>

- **Regression:** Predicts continuous numerical values. Example: Predicting temperature tomorrow (72.5 degrees)
- **Classification:** Predicts discrete categories. Example: Email spam detection (Spam vs Not Spam)

The key is output type: continuous numbers vs discrete categories.
</details>

---

### A2. Neural Network Theory (8 points)

**A2a.** (3 pts) Draw or describe the structure of a single perceptron. Label all components and write the mathematical formula.

<details>
<summary>Model Answer</summary>

**Components:**
1. **Inputs (x1, x2, ..., xn):** Data features
2. **Weights (w1, w2, ..., wn):** Learned parameters, one per input
3. **Bias (b):** Threshold/offset term
4. **Activation Function:** Non-linear transformation

**Formula:** output = activation(w1*x1 + w2*x2 + ... + wn*xn + b)

Or: a = f(sum(wi * xi) + b)
</details>

**A2b.** (3 pts) Why are activation functions necessary in neural networks? What happens if you remove them?

<details>
<summary>Model Answer</summary>

**Why necessary:** Activation functions introduce non-linearity, enabling neural networks to learn complex, non-linear patterns.

**Without them:** Multiple linear layers collapse into a single linear transformation, regardless of depth. The network can only learn linear decision boundaries.

Example: f(g(x)) where f and g are linear = f(Ax + b) = A'x + b' (still linear)
</details>

**A2c.** (2 pts) Compare ReLU and Sigmoid activation functions. When would you use each?

<details>
<summary>Model Answer</summary>

| Aspect | ReLU | Sigmoid |
|--------|------|---------|
| Formula | max(0, x) | 1/(1 + e^(-x)) |
| Range | [0, infinity) | (0, 1) |
| Gradient Issue | Dying ReLU | Vanishing gradients |
| Use Case | Hidden layers | Binary output layer |

- **ReLU:** Default for hidden layers - computationally efficient, avoids vanishing gradients
- **Sigmoid:** Output layer for binary classification - outputs interpretable probability
</details>

---

### A3. TensorFlow and Keras (8 points)

**A3a.** (2 pts) What is a tensor? Describe its three key properties.

<details>
<summary>Model Answer</summary>

A **tensor** is a multi-dimensional array of numbers with uniform data type.

**Key Properties:**
1. **Shape:** Dimensions of the tensor, e.g., (32, 224, 224, 3)
2. **Rank (ndim):** Number of dimensions (scalar=0, vector=1, matrix=2)
3. **Dtype:** Data type of elements (float32, int64, etc.)
</details>

**A3b.** (3 pts) What does the following code create? Explain each layer's purpose.

```python
model = Sequential([
    Dense(64, activation='relu', input_shape=(100,)),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])
```

<details>
<summary>Model Answer</summary>

This creates a **Multi-Layer Perceptron for 10-class classification**.

- **Layer 1 (Dense 64):** Input layer, receives 100 features, outputs 64 neurons with ReLU activation. Learns first-level representations.
- **Layer 2 (Dense 32):** Hidden layer, compresses 64 features to 32 with ReLU. Learns higher-level patterns.
- **Layer 3 (Dense 10):** Output layer, 10 neurons (one per class) with softmax to produce probability distribution.

Total parameters: (100*64 + 64) + (64*32 + 32) + (32*10 + 10) = 6,474 + 2,080 + 330 = 8,884
</details>

**A3c.** (3 pts) Explain what happens during `model.compile()` and why we need to call it before training.

<details>
<summary>Model Answer</summary>

`model.compile()` configures the model for training by specifying:

1. **Optimizer:** Algorithm for updating weights (e.g., 'adam', 'sgd')
2. **Loss Function:** How to measure prediction error (e.g., 'categorical_crossentropy')
3. **Metrics:** What to track during training (e.g., 'accuracy')

**Why necessary:** The model needs to know:
- How to adjust weights (optimizer + loss)
- What to report during training (metrics)

Without compile, `fit()` doesn't know how to compute gradients or update weights.
</details>

---

### A4. CNNs and Image Processing (8 points)

**A4a.** (3 pts) Explain how a convolutional layer processes an image. Include the concepts of filters, stride, and feature maps.

<details>
<summary>Model Answer</summary>

A convolutional layer applies **learnable filters** that slide across the input:

1. **Filter/Kernel:** Small matrix (e.g., 3x3) with learnable weights
2. **Convolution:** At each position, element-wise multiply filter with input patch, sum results
3. **Stride:** How many pixels to move filter (stride=1 moves one pixel at a time)
4. **Feature Map:** Output showing where features were detected

**Key benefits:**
- Local connectivity (each neuron sees small region)
- Weight sharing (same filter everywhere) - translation invariance
- Far fewer parameters than fully connected layers
</details>

**A4b.** (2 pts) What is pooling and why is it used in CNNs?

<details>
<summary>Model Answer</summary>

**Pooling** downsamples feature maps by taking the max or average value in each region.

**Purposes:**
1. **Reduces spatial dimensions** - fewer parameters, less computation
2. **Provides translation invariance** - slight shifts don't change output
3. **Highlights strongest features** (max pooling)

Example: 2x2 max pooling with stride 2: 28x28 becomes 14x14

**Note:** Pooling has no learnable parameters.
</details>

**A4c.** (3 pts) Why is flattening necessary before connecting CNN feature maps to Dense layers?

<details>
<summary>Model Answer</summary>

**CNN outputs:** 3D tensors with shape (height, width, channels), e.g., (7, 7, 512)

**Dense layer requires:** 1D vectors - each neuron connects to every input

**Flattening:** Reshapes (7, 7, 512) into (25,088) - a 1D vector

**Why necessary:** Dense layers expect 1D input. They can't process spatial data directly.

**Alternative:** GlobalAveragePooling2D creates smaller 1D output (7, 7, 512) -> (512) by averaging each channel.
</details>

---

## Section B: Practical Application (40 Points)

### B1. K-Means Clustering Implementation (10 points)

Given the following 2D data points, perform K-Means clustering with K=2 for ONE iteration.

**Points:** A(1, 2), B(2, 1), C(4, 5), D(5, 4)
**Initial Centroids:** C1(1, 1), C2(5, 5)

Use Euclidean distance.

**B1a.** (4 pts) Calculate the distance from each point to each centroid.

<details>
<summary>Model Answer</summary>

**Euclidean distance formula:** d = sqrt((x2-x1)^2 + (y2-y1)^2)

| Point | Distance to C1(1,1) | Distance to C2(5,5) |
|-------|---------------------|---------------------|
| A(1,2) | sqrt((1-1)^2 + (2-1)^2) = **1.0** | sqrt((1-5)^2 + (2-5)^2) = **5.0** |
| B(2,1) | sqrt((2-1)^2 + (1-1)^2) = **1.0** | sqrt((2-5)^2 + (1-5)^2) = **5.0** |
| C(4,5) | sqrt((4-1)^2 + (5-1)^2) = **5.0** | sqrt((4-5)^2 + (5-5)^2) = **1.0** |
| D(5,4) | sqrt((5-1)^2 + (4-1)^2) = **5.0** | sqrt((5-5)^2 + (4-5)^2) = **1.0** |
</details>

**B1b.** (3 pts) Assign each point to the nearest centroid.

<details>
<summary>Model Answer</summary>

**Assignment (closest centroid):**
- A(1,2) -> **Cluster 1** (distance 1.0 < 5.0)
- B(2,1) -> **Cluster 1** (distance 1.0 < 5.0)
- C(4,5) -> **Cluster 2** (distance 1.0 < 5.0)
- D(5,4) -> **Cluster 2** (distance 1.0 < 5.0)

**Result:**
- Cluster 1: {A, B}
- Cluster 2: {C, D}
</details>

**B1c.** (3 pts) Calculate the new centroids.

<details>
<summary>Model Answer</summary>

**New centroid = mean of assigned points**

**Cluster 1 (A, B):**
- C1_new_x = (1 + 2) / 2 = **1.5**
- C1_new_y = (2 + 1) / 2 = **1.5**
- **New C1 = (1.5, 1.5)**

**Cluster 2 (C, D):**
- C2_new_x = (4 + 5) / 2 = **4.5**
- C2_new_y = (5 + 4) / 2 = **4.5**
- **New C2 = (4.5, 4.5)**
</details>

---

### B2. Neural Network Forward Pass (15 points)

Consider a simple neural network with:
- 2 inputs
- 1 hidden layer with 2 neurons (ReLU activation)
- 1 output neuron (Sigmoid activation)

**Given weights and biases:**
```
Hidden layer:
  w1 = [0.5, 0.3]  (weights for neuron 1)
  w2 = [0.2, 0.4]  (weights for neuron 2)
  b_hidden = [0.1, -0.2]

Output layer:
  w_out = [0.6, 0.8]
  b_out = 0.3
```

**Input:** x = [2, 1]

**B2a.** (5 pts) Calculate the hidden layer output (before and after ReLU).

<details>
<summary>Model Answer</summary>

**Hidden neuron 1:**
z1 = w1[0]*x[0] + w1[1]*x[1] + b_hidden[0]
z1 = 0.5*2 + 0.3*1 + 0.1 = 1.0 + 0.3 + 0.1 = **1.4**
h1 = ReLU(1.4) = max(0, 1.4) = **1.4**

**Hidden neuron 2:**
z2 = w2[0]*x[0] + w2[1]*x[1] + b_hidden[1]
z2 = 0.2*2 + 0.4*1 + (-0.2) = 0.4 + 0.4 - 0.2 = **0.6**
h2 = ReLU(0.6) = max(0, 0.6) = **0.6**

**Hidden layer output: h = [1.4, 0.6]**
</details>

**B2b.** (5 pts) Calculate the final output (before and after Sigmoid).

<details>
<summary>Model Answer</summary>

**Output neuron:**
z_out = w_out[0]*h[0] + w_out[1]*h[1] + b_out
z_out = 0.6*1.4 + 0.8*0.6 + 0.3 = 0.84 + 0.48 + 0.3 = **1.62**

**Sigmoid:**
y = 1 / (1 + e^(-z_out))
y = 1 / (1 + e^(-1.62))
y = 1 / (1 + 0.198) = 1 / 1.198 = **0.835**

**Final output: 0.835** (approximately 83.5% probability for positive class)
</details>

**B2c.** (5 pts) If the true label is 1 and we use binary cross-entropy loss, calculate the loss.

<details>
<summary>Model Answer</summary>

**Binary Cross-Entropy formula:**
L = -[y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)]

**Given:**
- y_true = 1
- y_pred = 0.835

**Calculation:**
L = -[1 * log(0.835) + (1 - 1) * log(1 - 0.835)]
L = -[log(0.835) + 0]
L = -log(0.835)
L = -(-0.180)
L = **0.180**

Interpretation: The loss is relatively low because our prediction (0.835) is close to the true label (1).
</details>

---

### B3. CNN Architecture Analysis (15 points)

Analyze the following CNN architecture:

```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
```

**B3a.** (5 pts) Calculate the output shape after each layer.

<details>
<summary>Model Answer</summary>

| Layer | Input Shape | Output Shape | Calculation |
|-------|-------------|--------------|-------------|
| Input | - | (28, 28, 1) | Given |
| Conv2D(32) | (28, 28, 1) | **(26, 26, 32)** | 28-3+1=26 (no padding) |
| MaxPooling2D | (26, 26, 32) | **(13, 13, 32)** | 26/2=13 |
| Conv2D(64) | (13, 13, 32) | **(11, 11, 64)** | 13-3+1=11 |
| MaxPooling2D | (11, 11, 64) | **(5, 5, 64)** | 11/2=5 (floor) |
| Flatten | (5, 5, 64) | **(1600,)** | 5*5*64=1600 |
| Dense(128) | (1600,) | **(128,)** | Fully connected |
| Dense(10) | (128,) | **(10,)** | Output classes |
</details>

**B3b.** (5 pts) Calculate the number of trainable parameters in the first Conv2D layer and the first Dense layer.

<details>
<summary>Model Answer</summary>

**Conv2D(32, (3,3)) with input channels=1:**
- Weights: 3 * 3 * 1 * 32 = 288 (kernel_size * kernel_size * in_channels * filters)
- Biases: 32 (one per filter)
- **Total: 288 + 32 = 320 parameters**

**Dense(128) with input=1600:**
- Weights: 1600 * 128 = 204,800
- Biases: 128
- **Total: 204,800 + 128 = 204,928 parameters**

Note how the Dense layer has 640x more parameters than the Conv layer despite less "work" - this is why CNNs are parameter-efficient for images.
</details>

**B3c.** (5 pts) What problem does this architecture solve? How can you tell from the final layer?

<details>
<summary>Model Answer</summary>

**Problem:** 10-class image classification (e.g., MNIST digit recognition)

**How we can tell:**
1. **Input shape (28, 28, 1):** Grayscale 28x28 images - matches MNIST
2. **Final Dense(10, activation='softmax'):**
   - 10 neurons = 10 classes
   - Softmax = probability distribution over classes
   - Output sums to 1.0

**Architecture pattern:**
- Feature extraction (Conv layers) -> Spatial hierarchy learning
- Classification head (Dense layers) -> Decision making

The softmax activation with multiple outputs is the defining characteristic of multi-class classification.
</details>

---

## Section C: Debugging and Analysis (20 Points)

### C1. Training Curve Analysis (10 points)

Given these training metrics over 20 epochs:

```
Epoch 1:  train_loss=2.5, val_loss=2.6, train_acc=0.25, val_acc=0.23
Epoch 5:  train_loss=0.8, val_loss=0.9, train_acc=0.72, val_acc=0.70
Epoch 10: train_loss=0.3, val_loss=0.7, train_acc=0.91, val_acc=0.78
Epoch 15: train_loss=0.1, val_loss=1.2, train_acc=0.98, val_acc=0.75
Epoch 20: train_loss=0.05, val_loss=1.8, train_acc=0.99, val_acc=0.72
```

**C1a.** (4 pts) Describe what is happening to the model between epochs 5 and 20.

<details>
<summary>Model Answer</summary>

**The model is overfitting after epoch 5:**

**Evidence:**
- Training loss continues decreasing (0.8 -> 0.05)
- Validation loss increases after epoch 5 (0.9 -> 1.8)
- Training accuracy approaches 99% while validation accuracy drops from 78% to 72%
- Growing gap between train and validation metrics

**Pattern:** The model is memorizing training data rather than learning generalizable patterns. After epoch 5, additional training hurts generalization.
</details>

**C1b.** (3 pts) At which epoch should training have stopped? Justify your answer.

<details>
<summary>Model Answer</summary>

**Training should have stopped at epoch 5** (or possibly earlier between epochs 5-10).

**Justification:**
- Epoch 5: Best balance of train/val metrics (gap is smallest)
- Val_loss = 0.9 is the lowest (best generalization)
- Val_acc = 0.70 with train_acc = 0.72 (healthy small gap)
- After epoch 5: val_loss starts increasing (overfitting begins)

**Implementation:** Use EarlyStopping callback:
```python
EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
```
</details>

**C1c.** (3 pts) List three techniques to address this issue.

<details>
<summary>Model Answer</summary>

**Techniques to reduce overfitting:**

1. **Early Stopping:** Stop training when val_loss stops improving
   ```python
   EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
   ```

2. **Dropout:** Randomly zero neurons during training (regularization)
   ```python
   model.add(Dropout(0.5))
   ```

3. **Data Augmentation:** Create variations of training data
   ```python
   ImageDataGenerator(rotation_range=20, horizontal_flip=True)
   ```

**Additional options:** L2 regularization, reduce model complexity, collect more training data.
</details>

---

### C2. Code Debugging (10 points)

Identify and fix the errors in this code:

```python
# Task: Build a model for MNIST (28x28 grayscale images, 10 classes)
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Dense(128, activation='relu'),
    Dense(10, activation='relu')
])

model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(x_train, y_train, epochs=10)
```

**C2a.** (6 pts) Identify at least three errors in this code and explain why each is a problem.

<details>
<summary>Model Answer</summary>

**Error 1: Missing channel dimension in input_shape**
- `input_shape=(28, 28)` should be `input_shape=(28, 28, 1)`
- Conv2D expects 3D input (height, width, channels)
- MNIST is grayscale, so channels=1

**Error 2: Missing Flatten layer before Dense**
- Conv2D outputs 3D tensor (height, width, filters)
- Dense expects 1D input
- Need `Flatten()` layer between Conv and Dense

**Error 3: Wrong activation on output layer**
- `Dense(10, activation='relu')` should be `activation='softmax'`
- Softmax produces probability distribution for multi-class classification
- ReLU would produce unbounded positive outputs, not probabilities

**Error 4: Missing metrics in compile**
- Should include `metrics=['accuracy']` to monitor training
- Without it, only loss is shown during training
</details>

**C2b.** (4 pts) Write the corrected code.

<details>
<summary>Model Answer</summary>

```python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # Added channel dim
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),  # Added Flatten layer
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')  # Changed to softmax
])

model.compile(
    optimizer='adam', 
    loss='categorical_crossentropy',
    metrics=['accuracy']  # Added metrics
)

model.fit(x_train, y_train, epochs=10, validation_split=0.2)  # Added validation
```
</details>

---

## Section D: Short Essay (10 Points)

Choose ONE of the following questions. Write a clear, structured response (150-250 words).

### Option 1: The Role of Feature Hierarchies in Deep Learning

Explain how neural networks learn hierarchical representations. Use a CNN processing an image of a face as a concrete example. Discuss what each layer might learn.

<details>
<summary>Model Answer</summary>

Neural networks learn **hierarchical features** where each layer builds on the previous layer's outputs, creating increasingly abstract representations.

**Example: CNN Processing a Face**

**Early Layers (Low-level Features):**
The first convolutional layers learn to detect simple patterns:
- Edges at various orientations
- Color gradients and contrasts
- Basic textures
These are generic features useful for any image.

**Middle Layers (Mid-level Features):**
Subsequent layers combine low-level features into parts:
- Edges combine into curves
- Curves and textures form shapes like eyes, nose, mouth
- Color patterns form skin tones

**Deep Layers (High-level Features):**
Final convolutional layers recognize complex patterns:
- Individual facial features as complete units
- Spatial relationships between features
- Identity-specific characteristics

**Dense Layers (Abstract Concepts):**
The classification head combines spatial features into identity judgment.

**Key Insight:** No one programs "look for eyes." The network discovers that detecting eye-like patterns is useful for distinguishing faces - this is representation learning. Each layer transforms data into representations more suitable for the final task.
</details>

---

### Option 2: Comparing Learning Paradigms

Compare supervised and unsupervised learning from a practical perspective. Discuss when a data scientist would choose each approach, what challenges each presents, and how they might be combined.

<details>
<summary>Model Answer</summary>

**Supervised Learning** uses labeled data to learn input-output mappings. It's chosen when:
- Clear prediction targets exist (spam detection, price prediction)
- Labeled data is available
- Performance can be measured against known answers

**Challenges:** Requires expensive labeling, limited by label quality, can't discover unknown patterns.

**Unsupervised Learning** finds patterns in unlabeled data. It's chosen when:
- No labels exist or are too expensive
- Goal is discovery (customer segments, anomalies)
- Understanding data structure is primary

**Challenges:** Hard to evaluate (no "correct" answer), results need interpretation, may find meaningless patterns.

**Combining Approaches:**
1. **Pre-training:** Use unsupervised learning to learn representations, then fine-tune with limited labels (transfer learning)
2. **Semi-supervised:** Cluster unlabeled data, use clusters to augment labeled dataset
3. **Feature engineering:** Use clustering to create features for supervised models

**Example:** Cluster customers (unsupervised), then build purchase prediction model (supervised) using cluster membership as a feature.
</details>

---

## Answer Key Summary

| Section | Points | Passing (70%) |
|---------|--------|---------------|
| A: Conceptual | 30 | 21 |
| B: Practical | 40 | 28 |
| C: Debugging | 20 | 14 |
| D: Essay | 10 | 7 |
| **Total** | **100** | **70** |

---

*Weekly Technical Assessment generated by Quality Assurance Agent*
*Week 1: AI/ML Fundamentals - From Zero to Neural*

