# Tensors and Shapes

## Learning Objectives

- Understand tensors as the fundamental data structure in TensorFlow
- Identify scalar, vector, matrix, and higher-rank tensors
- Master shape notation and interpret tensor dimensions
- Apply broadcasting rules and common shape manipulation operations

## Why This Matters

Every piece of data in TensorFlow - your input features, weights, activations, and predictions - is represented as a tensor. Understanding tensor shapes is essential because:

1. **Shape mismatches** are the most common source of neural network bugs
2. **Efficient batching** requires understanding how shapes propagate
3. **Debugging** requires interpreting shape error messages

In our **From Zero to Neural** journey, tensors are the language TensorFlow speaks. Master shapes, and you'll build networks with confidence.

## The Concept

### What Is a Tensor?

A **tensor** is a multi-dimensional array of numbers with a uniform data type. Tensors generalize scalars, vectors, and matrices to arbitrary dimensions.

```
Tensor Hierarchy:

Rank 0 (Scalar):     5
                     (single number)

Rank 1 (Vector):     [1, 2, 3]
                     (1D array)

Rank 2 (Matrix):     [[1, 2],
                      [3, 4],
                      [5, 6]]
                     (2D array)

Rank 3:              [[[1,2], [3,4]],
                      [[5,6], [7,8]]]
                     (3D array, e.g., batch of images)

Rank n:              ... (n-dimensional array)
```

### Tensor Properties

Every tensor has three key properties:

| Property | Description | Example |
|----------|-------------|---------|
| **Shape** | Dimensions of the tensor | (3, 4, 5) |
| **Rank** | Number of dimensions | 3 |
| **Dtype** | Data type | float32, int64 |

```python
import tensorflow as tf

t = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=tf.float32)

print(f"Tensor:\n{t}")
print(f"Shape: {t.shape}")      # (2, 2, 2)
print(f"Rank: {len(t.shape)}")  # 3
print(f"Dtype: {t.dtype}")      # float32
```

### Understanding Shape Notation

Shape is represented as a tuple of integers, one per dimension:

```
Shape: (batch_size, height, width, channels)

Example: (32, 224, 224, 3)
         |   |    |    |
         |   |    |    +-- 3 color channels (RGB)
         |   |    +------- 224 pixels wide
         |   +------------ 224 pixels tall
         +---------------- 32 images in batch
```

**Common Shape Patterns:**

| Shape | Meaning | Example Use |
|-------|---------|-------------|
| () | Scalar | Loss value |
| (n,) | Vector of n elements | 1D features |
| (m, n) | m rows x n columns | Dense layer weights |
| (batch, features) | Batch of feature vectors | Network input |
| (batch, height, width, channels) | Batch of images | CNN input |
| (batch, timesteps, features) | Batch of sequences | RNN input |

### Tensor Ranks in Practice

**Rank 0: Scalar**
```python
scalar = tf.constant(42)
print(f"Scalar: {scalar}")
print(f"Shape: {scalar.shape}")  # ()
```

Use cases: Loss values, single predictions, constants

**Rank 1: Vector**
```python
vector = tf.constant([1.0, 2.0, 3.0, 4.0])
print(f"Vector: {vector}")
print(f"Shape: {vector.shape}")  # (4,)
```

Use cases: Bias terms, 1D features, class probabilities

**Rank 2: Matrix**
```python
matrix = tf.constant([[1, 2, 3],
                      [4, 5, 6]])
print(f"Matrix:\n{matrix}")
print(f"Shape: {matrix.shape}")  # (2, 3)
```

Use cases: Weight matrices, batches of 1D data, grayscale images

**Rank 3: 3D Tensor**
```python
tensor_3d = tf.constant([[[1, 2], [3, 4]],
                         [[5, 6], [7, 8]],
                         [[9, 10], [11, 12]]])
print(f"3D Tensor shape: {tensor_3d.shape}")  # (3, 2, 2)
```

Use cases: Batch of matrices, sequences of vectors, RGB images

**Rank 4: 4D Tensor**
```python
# Batch of images: (batch, height, width, channels)
batch_images = tf.random.normal([32, 28, 28, 1])  # 32 grayscale 28x28 images
print(f"Batch shape: {batch_images.shape}")  # (32, 28, 28, 1)
```

Use cases: Batches of images (standard CNN input format)

### Shape Manipulation Operations

**Reshape: Change shape without changing data**
```python
t = tf.constant([1, 2, 3, 4, 5, 6])
print(f"Original: {t.shape}")  # (6,)

# Reshape to 2x3 matrix
reshaped = tf.reshape(t, [2, 3])
print(f"Reshaped:\n{reshaped}")  # [[1,2,3], [4,5,6]]
print(f"New shape: {reshaped.shape}")  # (2, 3)

# Use -1 to infer dimension
auto_reshaped = tf.reshape(t, [3, -1])  # TF figures out it's 3x2
print(f"Auto reshaped: {auto_reshaped.shape}")  # (3, 2)
```

**Expand Dims: Add a dimension**
```python
t = tf.constant([1, 2, 3])
print(f"Original: {t.shape}")  # (3,)

# Add dimension at axis 0 (batch dimension)
expanded = tf.expand_dims(t, axis=0)
print(f"Expanded at axis 0: {expanded.shape}")  # (1, 3)

# Add dimension at axis 1
expanded = tf.expand_dims(t, axis=1)
print(f"Expanded at axis 1: {expanded.shape}")  # (3, 1)
```

**Squeeze: Remove dimensions of size 1**
```python
t = tf.constant([[[1, 2, 3]]])  # Shape (1, 1, 3)
squeezed = tf.squeeze(t)
print(f"Squeezed: {squeezed.shape}")  # (3,)

# Squeeze specific axis
t = tf.constant([[[1], [2], [3]]])  # Shape (1, 3, 1)
squeezed_axis = tf.squeeze(t, axis=0)
print(f"Squeezed axis 0: {squeezed_axis.shape}")  # (3, 1)
```

**Transpose: Swap dimensions**
```python
matrix = tf.constant([[1, 2, 3],
                      [4, 5, 6]])  # Shape (2, 3)
transposed = tf.transpose(matrix)
print(f"Transposed:\n{transposed}")  # Shape (3, 2)
```

### Broadcasting Rules

Broadcasting allows operations between tensors of different shapes by automatically expanding dimensions.

**Rule 1**: Dimensions are compared right-to-left
**Rule 2**: Dimensions match if equal or one is 1
**Rule 3**: Missing dimensions are treated as 1

```
Broadcasting Examples:

Shape (3, 4) + Shape (4,) = Shape (3, 4)
    [3, 4]     [-, 4]   
    
Shape (3, 1) + Shape (1, 4) = Shape (3, 4)
    [3, 1]     [1, 4]

Shape (2, 3, 4) + Shape (3, 4) = Shape (2, 3, 4)
```

**Code Example:**
```python
# Broadcasting in action
matrix = tf.constant([[1, 2, 3],
                      [4, 5, 6]])  # Shape (2, 3)
vector = tf.constant([10, 20, 30])  # Shape (3,)

# Vector is broadcast to match matrix shape
result = matrix + vector
print(f"Matrix + Vector:\n{result}")
# [[11, 22, 33],
#  [14, 25, 36]]

# Column vector broadcast
col_vector = tf.constant([[100], [200]])  # Shape (2, 1)
result = matrix + col_vector
print(f"Matrix + Column:\n{result}")
# [[101, 102, 103],
#  [204, 205, 206]]
```

### Common Shape Errors and Fixes

**Error 1: Matrix multiplication shape mismatch**
```python
A = tf.constant([[1, 2, 3]])      # Shape (1, 3)
B = tf.constant([[1, 2], [3, 4]]) # Shape (2, 2)

# This fails: inner dimensions must match (3 != 2)
# C = A @ B  # Error!

# Fix: transpose or reshape
B_correct = tf.constant([[1, 2], [3, 4], [5, 6]])  # Shape (3, 2)
C = A @ B_correct  # Works: (1, 3) @ (3, 2) = (1, 2)
```

**Error 2: Unexpected batch dimension**
```python
# Model expects (batch, features)
model_input = tf.constant([1.0, 2.0, 3.0])  # Shape (3,) - no batch!

# Fix: add batch dimension
model_input = tf.expand_dims(model_input, axis=0)  # Shape (1, 3)
```

**Error 3: Channel dimension mismatch**
```python
# CNN expects (batch, height, width, channels)
image = tf.random.normal([28, 28])  # Shape (28, 28) - missing batch and channel!

# Fix: add dimensions
image = tf.expand_dims(image, axis=0)   # (1, 28, 28)
image = tf.expand_dims(image, axis=-1)  # (1, 28, 28, 1)
```

## Code Example: Complete Shape Operations

```python
import tensorflow as tf
import numpy as np

print("=" * 60)
print("TENSOR SHAPES DEEP DIVE")
print("=" * 60)

# === Creating Tensors of Different Ranks ===
print("\n--- Tensor Ranks ---")

scalar = tf.constant(42)
print(f"Scalar: value={scalar.numpy()}, shape={scalar.shape}, rank={len(scalar.shape)}")

vector = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0])
print(f"Vector: shape={vector.shape}, rank={len(vector.shape)}")

matrix = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"Matrix: shape={matrix.shape}, rank={len(matrix.shape)}")

tensor_3d = tf.random.normal([2, 3, 4])
print(f"3D Tensor: shape={tensor_3d.shape}, rank={len(tensor_3d.shape)}")

# === Neural Network Shape Flow ===
print("\n--- Neural Network Shape Example ---")

batch_size = 32
input_features = 784  # 28x28 flattened

# Simulated data batch
X = tf.random.normal([batch_size, input_features])
print(f"Input batch: {X.shape}")

# Dense layer: 784 -> 128
W1 = tf.Variable(tf.random.normal([input_features, 128]) * 0.01)
b1 = tf.Variable(tf.zeros([128]))
print(f"Layer 1 weights: {W1.shape}, bias: {b1.shape}")

# Forward through layer 1
Z1 = X @ W1 + b1
A1 = tf.nn.relu(Z1)
print(f"After layer 1: {A1.shape}")

# Dense layer: 128 -> 10
W2 = tf.Variable(tf.random.normal([128, 10]) * 0.01)
b2 = tf.Variable(tf.zeros([10]))
print(f"Layer 2 weights: {W2.shape}, bias: {b2.shape}")

# Forward through layer 2
Z2 = A1 @ W2 + b2
output = tf.nn.softmax(Z2)
print(f"Output (predictions): {output.shape}")

# === Shape Manipulation ===
print("\n--- Shape Manipulation ---")

original = tf.range(24)
print(f"Original: shape={original.shape}")

# Various reshapes
reshape_2d = tf.reshape(original, [4, 6])
reshape_3d = tf.reshape(original, [2, 3, 4])
reshape_4d = tf.reshape(original, [2, 2, 2, 3])

print(f"Reshaped to 2D: {reshape_2d.shape}")
print(f"Reshaped to 3D: {reshape_3d.shape}")
print(f"Reshaped to 4D: {reshape_4d.shape}")

# Using -1 for automatic dimension
auto_reshape = tf.reshape(original, [4, -1])
print(f"Auto reshape [4, -1]: {auto_reshape.shape}")

# === Expand and Squeeze ===
print("\n--- Expand and Squeeze ---")

features = tf.constant([1.0, 2.0, 3.0, 4.0])
print(f"Original features: {features.shape}")

# Add batch dimension
batched = tf.expand_dims(features, axis=0)
print(f"With batch dim: {batched.shape}")

# Add channel dimension
with_channel = tf.expand_dims(batched, axis=-1)
print(f"With channel dim: {with_channel.shape}")

# Squeeze removes size-1 dimensions
squeezed = tf.squeeze(with_channel)
print(f"After squeeze: {squeezed.shape}")

# === Broadcasting ===
print("\n--- Broadcasting ---")

# Matrix (3, 4) + Vector (4,)
mat = tf.ones([3, 4])
vec = tf.constant([1.0, 2.0, 3.0, 4.0])
result = mat + vec
print(f"Matrix {mat.shape} + Vector {vec.shape} = {result.shape}")

# Batch broadcasting
batch = tf.ones([32, 10])
bias = tf.constant([0.1] * 10)  # Shape (10,)
result = batch + bias  # Broadcasts to (32, 10)
print(f"Batch {batch.shape} + Bias {bias.shape} = {result.shape}")

# === Common Patterns ===
print("\n--- Common Patterns ---")

# Image batch: (batch, height, width, channels)
images = tf.random.normal([16, 224, 224, 3])
print(f"Image batch: {images.shape}")
print(f"  - {images.shape[0]} images")
print(f"  - {images.shape[1]}x{images.shape[2]} resolution")
print(f"  - {images.shape[3]} channels (RGB)")

# Sequence batch: (batch, timesteps, features)
sequences = tf.random.normal([32, 100, 256])
print(f"Sequence batch: {sequences.shape}")
print(f"  - {sequences.shape[0]} sequences")
print(f"  - {sequences.shape[1]} timesteps")
print(f"  - {sequences.shape[2]} features per timestep")

print("\n" + "=" * 60)
```

## Key Takeaways

1. **Tensors are multi-dimensional arrays** - the fundamental data structure in TensorFlow.

2. **Shape is crucial** - (batch, features), (batch, height, width, channels) conventions matter.

3. **Rank = number of dimensions** - scalars (0), vectors (1), matrices (2), and beyond.

4. **Reshape, expand_dims, squeeze** - essential tools for shape manipulation.

5. **Broadcasting automatically expands shapes** - understand the rules to avoid surprises.

## Looking Ahead

With tensor shapes mastered, the next reading on **Graphs and Operations** explains how TensorFlow actually executes computations on these tensors. Then we'll use this knowledge to build models with the **Keras Sequential API**.

## Additional Resources

- [TensorFlow Tensors Guide](https://www.tensorflow.org/guide/tensor) - Official documentation
- [NumPy Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) - Rules apply to TensorFlow too
- [Shape Debugging Tips](https://www.tensorflow.org/guide/intro_to_modules) - Common issues and solutions

