# Exercise 07: Tensor Manipulation Lab

## Learning Objectives

- Create tensors from various sources
- Manipulate tensor shapes confidently
- Apply broadcasting rules
- Debug common shape errors

## Duration

**Estimated Time:** 60 minutes

---

## Part 1: Tensor Creation (15 min)

### Task 1.1: Create Tensors

Navigate to `starter_code/tensor_lab.py`:

```python
import tensorflow as tf
import numpy as np

# TODO: Create these tensors

# 1. A scalar (rank 0) with value 42
scalar = None

# 2. A vector (rank 1) with values [1, 2, 3, 4, 5]
vector = None

# 3. A 3x3 matrix (rank 2) of all ones
matrix_ones = None

# 4. A 2x3x4 tensor (rank 3) of zeros
tensor_3d = None

# 5. A tensor from a NumPy array
np_array = np.array([[1, 2], [3, 4], [5, 6]])
from_numpy = None

# 6. A random normal tensor with shape (100, 10), mean=0, stddev=1
random_normal = None

# 7. A tensor with values from 0 to 99
range_tensor = None

# Print shapes and dtypes for each
```

### Task 1.2: Verify Properties

For each tensor, print:
- Shape
- Dtype
- Rank (number of dimensions)
- Total number of elements

---

## Part 2: Shape Manipulation (20 min)

### Task 2.1: Reshape Operations

```python
# Start with this tensor
original = tf.range(24)
print(f"Original: {original.shape}")

# TODO: Reshape to each of these shapes
# shape_a = (4, 6)
# shape_b = (2, 3, 4)
# shape_c = (24, 1)
# shape_d = (1, 24)
# shape_e = (2, 2, 2, 3)

# Use tf.reshape(original, shape)
```

### Task 2.2: Using -1 for Automatic Dimension

```python
# tf.reshape can infer one dimension with -1

# TODO: Reshape original to (6, -1). What does -1 become?
reshape_auto = tf.reshape(original, [6, -1])
print(f"Shape with -1: {reshape_auto.shape}")

# TODO: Try (8, -1). What happens?
# reshape_invalid = tf.reshape(original, [8, -1])  # Will this work?
```

### Task 2.3: Expand and Squeeze

```python
vector = tf.constant([1, 2, 3, 4])

# TODO: Add a dimension at position 0
# Result should be shape (1, 4)
expanded_0 = None

# TODO: Add a dimension at position 1
# Result should be shape (4, 1)
expanded_1 = None

# TODO: Remove the extra dimension
squeezed = tf.squeeze(expanded_0)
print(f"Squeezed: {squeezed.shape}")
```

---

## Part 3: Broadcasting (15 min)

### Task 3.1: Basic Broadcasting

```python
# Broadcasting allows operations on different shapes

a = tf.constant([[1, 2, 3],
                 [4, 5, 6]])  # Shape: (2, 3)

b = tf.constant([10, 20, 30])  # Shape: (3,)

# TODO: Add a and b. What is the result shape?
result = None
print(f"Result shape: {result.shape}")
print(f"Result:\n{result}")

# TODO: Explain how broadcasting worked here
# Answer:
```

### Task 3.2: Broadcast Scenarios

```python
# Predict the result shape before running!

# Scenario 1
a1 = tf.ones([3, 1])
b1 = tf.ones([1, 4])
# Predicted result shape: ???
# c1 = a1 + b1

# Scenario 2
a2 = tf.ones([5, 3, 1])
b2 = tf.ones([1, 4])
# Predicted result shape: ???
# c2 = a2 + b2

# Scenario 3 (Will this work?)
a3 = tf.ones([3, 4])
b3 = tf.ones([3, 5])
# c3 = a3 + b3  # ???
```

---

## Part 4: Common Operations (10 min)

### Task 4.1: Math Operations

```python
x = tf.constant([[1., 2., 3.],
                 [4., 5., 6.]])

# TODO: Calculate each of these
mean_all = None       # Mean of all elements
mean_rows = None      # Mean of each row (axis=1)
mean_cols = None      # Mean of each column (axis=0)
sum_all = None        # Sum of all elements
max_val = None        # Maximum value
argmax_rows = None    # Index of max in each row
```

### Task 4.2: Matrix Operations

```python
A = tf.constant([[1., 2.],
                 [3., 4.]])

B = tf.constant([[5., 6.],
                 [7., 8.]])

# TODO: Matrix multiplication (not element-wise!)
matmul_result = None  # tf.matmul or @

# TODO: Element-wise multiplication
elementwise = None

# TODO: Transpose
transposed = None
```

---

## Part 5: Challenge Problems

### Challenge A: Batch Normalization Prep

Normalize a batch of images (shape: batch, height, width, channels) so each channel has mean 0 and std 1:

```python
# Fake image batch (2 images, 4x4, 3 channels)
images = tf.random.normal([2, 4, 4, 3])

# TODO: Calculate mean and std for each channel
# Normalize: (images - mean) / std
```

### Challenge B: Softmax from Scratch

Implement softmax without using tf.nn.softmax:

```python
def manual_softmax(logits):
    """
    Softmax: exp(x_i) / sum(exp(x_j))
    
    Hint: Subtract max for numerical stability
    """
    # TODO: Implement
    pass

# Test
logits = tf.constant([2.0, 1.0, 0.1])
# Result should sum to 1.0
```

### Challenge C: Shape Debugging

Fix these shape errors:

```python
# Error 1: Matrix multiply shape mismatch
a = tf.ones([3, 4])
b = tf.ones([5, 6])
# c = tf.matmul(a, b)  # This fails. How to fix?

# Error 2: Broadcasting failure
x = tf.ones([3, 4, 5])
y = tf.ones([4, 6])
# z = x + y  # This fails. How to fix?
```

---

## Definition of Done

- [ ] All tensors created with correct shapes
- [ ] Reshape operations completed
- [ ] Broadcasting exercises completed with explanations
- [ ] Math operations verified
- [ ] At least 1 challenge problem attempted

---

## Key Concepts

1. **Shapes matter!** Most bugs in deep learning are shape mismatches
2. **Broadcasting rules:** Dimensions are compared right-to-left
3. **-1 in reshape:** TensorFlow infers the size
4. **expand_dims vs squeeze:** Add or remove dimensions of size 1

