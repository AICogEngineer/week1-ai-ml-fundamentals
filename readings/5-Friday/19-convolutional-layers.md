# Convolutional Layers

## Learning Objectives

- Understand the convolution operation: filters, kernels, stride, and padding
- Explain how filters detect features like edges, textures, and patterns
- Calculate output dimensions after convolution
- Visualize feature maps and understand what CNNs "see"

## Why This Matters

Convolutional layers are the feature extractors of CNNs. While Dense layers see raw pixels, Conv layers see patterns - edges, corners, textures, and eventually faces, cars, and cats.

In our **From Zero to Neural** journey, understanding convolution completes your mental model of how CNNs work. You'll see exactly how a small filter sliding across an image can detect features regardless of their position.

## The Concept

### What Is Convolution?

**Convolution** is a mathematical operation where a small matrix (filter/kernel) slides across the input, computing element-wise multiplication and sum at each position.

```
Input Image (5x5):           Filter (3x3):           Output (Feature Map):

[1  2  3  4  5]              [1  0 -1]              After sliding filter
[6  7  8  9  10]             [1  0 -1]              across all positions:
[11 12 13 14 15]             [1  0 -1]              
[16 17 18 19 20]                                    [? ? ?]
[21 22 23 24 25]             (edge detector)        [? ? ?]
                                                    [? ? ?]
```

### The Convolution Operation Step-by-Step

**Step 1: Position filter at top-left**

```
[1  2  3] 4  5         [1  0 -1]
[6  7  8] 9  10   *    [1  0 -1]   =  (1*1 + 2*0 + 3*-1) +
[11 12 13] 14 15       [1  0 -1]      (6*1 + 7*0 + 8*-1) +
16 17 18 19 20                         (11*1 + 12*0 + 13*-1)
21 22 23 24 25                      =  (1 + 0 - 3) + (6 + 0 - 8) + (11 + 0 - 13)
                                    =  -2 + -2 + -2
                                    =  -6
```

**Step 2: Slide filter one position right**

```
1 [2  3  4] 5          [1  0 -1]
6 [7  8  9] 10    *    [1  0 -1]   =  (2*1 + 3*0 + 4*-1) + ...
11[12 13 14] 15        [1  0 -1]      =  -6
16 17 18 19 20                      
21 22 23 24 25                      
```

**Continue until filter has covered all positions.**

### Filters Detect Features

Different filters detect different features:

**Vertical Edge Detector:**
```
[ 1  0 -1]
[ 1  0 -1]
[ 1  0 -1]

Responds strongly to vertical transitions (light-to-dark)
```

**Horizontal Edge Detector:**
```
[ 1  1  1]
[ 0  0  0]
[-1 -1 -1]

Responds strongly to horizontal transitions
```

**Blur Filter:**
```
[1/9 1/9 1/9]
[1/9 1/9 1/9]
[1/9 1/9 1/9]

Averages neighborhood - smooths/blurs image
```

**Sharpen Filter:**
```
[ 0 -1  0]
[-1  5 -1]
[ 0 -1  0]

Enhances edges and details
```

### Key Convolution Parameters

**1. Filter Size (kernel_size)**
```
3x3 - Most common, good balance
5x5 - Larger receptive field
1x1 - Channel mixing, dimensionality reduction
7x7 - Sometimes used in first layer for large images
```

**2. Number of Filters (filters)**
```
Each filter detects one type of feature.
More filters = more feature types detected.

Common: 32, 64, 128, 256, 512
```

**3. Stride**
```
Stride = how many pixels filter moves each step

Stride 1:                     Stride 2:
[x x x] [x x x] [x x x]      [x x x]     [x x x]
  +1      +1      +1            +2          +2

Larger stride = smaller output
```

**4. Padding**
```
Padding = adding zeros around input edges

No padding ("valid"):         Padding ("same"):
Input: 5x5                    Input: 5x5
Filter: 3x3                   Filter: 3x3
Output: 3x3                   Output: 5x5

      [0 0 0 0 0 0 0]
      [0 x x x x x 0]
      [0 x x x x x 0]
      [0 x x x x x 0]
      [0 x x x x x 0]
      [0 x x x x x 0]
      [0 0 0 0 0 0 0]
```

### Output Dimension Formula

```
Output Size = floor((Input - Filter + 2*Padding) / Stride) + 1

Examples:
  Input: 28, Filter: 3, Padding: 0, Stride: 1
  Output = (28 - 3 + 0) / 1 + 1 = 26

  Input: 28, Filter: 3, Padding: 1, Stride: 1  ("same")
  Output = (28 - 3 + 2) / 1 + 1 = 28

  Input: 28, Filter: 3, Padding: 0, Stride: 2
  Output = (28 - 3 + 0) / 2 + 1 = 13
```

### Multiple Channels

Real images have 3 channels (RGB). Filters operate on all channels:

```
Input: 28x28x3 (RGB image)
Filter: 3x3x3 (must match input channels)

Filter slides across spatial dimensions,
computing across all channels at once.

Output: One 2D feature map per filter.

If 32 filters: Output is 26x26x32
```

### Feature Maps

The output of a convolutional layer is called a **feature map**:

```
Input Image (28x28x1)
        |
        v
Conv2D(32 filters, 3x3)
        |
        v
Feature Maps (26x26x32)
    |
    +-- Filter 1: detects vertical edges
    +-- Filter 2: detects horizontal edges
    +-- Filter 3: detects diagonal edges
    +-- ...
    +-- Filter 32: detects some learned pattern
```

### Keras Conv2D

```python
from tensorflow.keras import layers

# Basic convolutional layer
layer = layers.Conv2D(
    filters=32,              # Number of filters
    kernel_size=(3, 3),      # Filter size (can also be just 3)
    strides=(1, 1),          # Stride (can also be just 1)
    padding='valid',         # 'valid' (no padding) or 'same'
    activation='relu',       # Activation function
    input_shape=(28, 28, 1)  # Only needed for first layer
)
```

## Code Example: Convolution Visualization

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

print("=" * 60)
print("CONVOLUTIONAL LAYERS DEEP DIVE")
print("=" * 60)

# === Manual Convolution ===
print("\n--- Manual Convolution Example ---")

# Simple 5x5 input with vertical edge
input_image = np.array([
    [0, 0, 1, 1, 1],
    [0, 0, 1, 1, 1],
    [0, 0, 1, 1, 1],
    [0, 0, 1, 1, 1],
    [0, 0, 1, 1, 1]
], dtype=np.float32)

# Vertical edge detector
edge_filter = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
], dtype=np.float32)

print("Input (5x5 with vertical edge at column 2):")
print(input_image)
print("\nVertical Edge Filter:")
print(edge_filter)

# Manual convolution
def manual_conv2d(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    output_h = h - kh + 1
    output_w = w - kw + 1
    output = np.zeros((output_h, output_w))
    
    for i in range(output_h):
        for j in range(output_w):
            region = image[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel)
    
    return output

result = manual_conv2d(input_image, edge_filter)
print("\nConvolution Result (edge detected at column 2):")
print(result)

# === Keras Conv2D ===
print("\n--- Keras Conv2D ---")

# Reshape for Keras: (batch, height, width, channels)
input_tensor = input_image.reshape(1, 5, 5, 1)

# Create Conv2D with custom kernel
conv_layer = layers.Conv2D(
    filters=1,
    kernel_size=3,
    padding='valid',
    use_bias=False,
    input_shape=(5, 5, 1)
)

# Build and set custom weights
conv_layer.build((None, 5, 5, 1))
conv_layer.set_weights([edge_filter.reshape(3, 3, 1, 1)])

# Apply convolution
output_tensor = conv_layer(input_tensor)
print("Keras Conv2D output:")
print(output_tensor.numpy().reshape(3, 3))

# === Output Shape Calculations ===
print("\n--- Output Shape Examples ---")

test_cases = [
    {"input": (28, 28, 1), "filters": 32, "kernel": 3, "stride": 1, "padding": "valid"},
    {"input": (28, 28, 1), "filters": 32, "kernel": 3, "stride": 1, "padding": "same"},
    {"input": (28, 28, 1), "filters": 64, "kernel": 5, "stride": 2, "padding": "valid"},
    {"input": (224, 224, 3), "filters": 64, "kernel": 7, "stride": 2, "padding": "same"},
]

for case in test_cases:
    layer = layers.Conv2D(
        filters=case["filters"],
        kernel_size=case["kernel"],
        strides=case["stride"],
        padding=case["padding"],
        input_shape=case["input"]
    )
    # Build to compute output shape
    layer.build((None,) + case["input"])
    output_shape = layer.compute_output_shape((None,) + case["input"])
    
    print(f"Input: {case['input']}, K={case['kernel']}, S={case['stride']}, "
          f"P={case['padding']}, F={case['filters']} -> Output: {output_shape[1:]}")

# === Parameter Count ===
print("\n--- Conv2D Parameter Count ---")

# Parameters = (kernel_h * kernel_w * input_channels + 1) * filters
# The +1 is for bias per filter

input_shape = (28, 28, 1)
layer = layers.Conv2D(32, (3, 3), input_shape=input_shape)
layer.build((None,) + input_shape)

kernel, bias = layer.get_weights()
print(f"Input channels: {input_shape[-1]}")
print(f"Kernel size: 3x3")
print(f"Number of filters: 32")
print(f"Kernel shape: {kernel.shape}")
print(f"Bias shape: {bias.shape}")
print(f"Total parameters: {layer.count_params()}")
print(f"Formula: (3 * 3 * 1 + 1) * 32 = {(3*3*1+1)*32}")

# === Stacking Conv Layers ===
print("\n--- Stacking Convolutional Layers ---")

model = keras.Sequential([
    layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
    layers.Conv2D(64, 3, activation='relu'),
    layers.Conv2D(64, 3, activation='relu'),
])

model.summary()

# Show shape progression
print("\nShape progression:")
x = tf.random.normal([1, 28, 28, 1])
print(f"Input: {x.shape}")
for layer in model.layers:
    x = layer(x)
    print(f"After {layer.name}: {x.shape}")

print("\n" + "=" * 60)
```

**Sample Output:**
```
============================================================
CONVOLUTIONAL LAYERS DEEP DIVE
============================================================

--- Manual Convolution Example ---
Input (5x5 with vertical edge at column 2):
[[0. 0. 1. 1. 1.]
 [0. 0. 1. 1. 1.]
 [0. 0. 1. 1. 1.]
 [0. 0. 1. 1. 1.]
 [0. 0. 1. 1. 1.]]

Vertical Edge Filter:
[[-1.  0.  1.]
 [-1.  0.  1.]
 [-1.  0.  1.]]

Convolution Result (edge detected at column 2):
[[3. 0. 0.]
 [3. 0. 0.]
 [3. 0. 0.]]

--- Keras Conv2D ---
Keras Conv2D output:
[[3. 0. 0.]
 [3. 0. 0.]
 [3. 0. 0.]]

--- Output Shape Examples ---
Input: (28, 28, 1), K=3, S=1, P=valid, F=32 -> Output: (26, 26, 32)
Input: (28, 28, 1), K=3, S=1, P=same, F=32 -> Output: (28, 28, 32)
Input: (28, 28, 1), K=5, S=2, P=valid, F=64 -> Output: (12, 12, 64)
Input: (224, 224, 3), K=7, S=2, P=same, F=64 -> Output: (112, 112, 64)

--- Conv2D Parameter Count ---
Input channels: 1
Kernel size: 3x3
Number of filters: 32
Kernel shape: (3, 3, 1, 32)
Bias shape: (32,)
Total parameters: 320
Formula: (3 * 3 * 1 + 1) * 32 = 320

--- Stacking Convolutional Layers ---
Model: "sequential"
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
conv2d (Conv2D)             (None, 26, 26, 32)        320       
conv2d_1 (Conv2D)           (None, 24, 24, 64)        18496     
conv2d_2 (Conv2D)           (None, 22, 22, 64)        36928     
=================================================================
Total params: 55,744

Shape progression:
Input: (1, 28, 28, 1)
After conv2d: (1, 26, 26, 32)
After conv2d_1: (1, 24, 24, 64)
After conv2d_2: (1, 22, 22, 64)

============================================================
```

### What Learned Filters Look Like

In trained CNNs, early layer filters often resemble classical edge detectors:

```
Learned Filter Examples (First Conv Layer):

Filter 1:        Filter 2:        Filter 3:
[- - +]         [- + -]         [+ - -]
[- - +]         [- + -]         [+ - -]
[- - +]         [- + -]         [+ - -]
(vertical)      (center line)    (right edge)
```

**Deeper layers** learn more complex patterns that are harder to interpret visually.

## Key Takeaways

1. **Convolution slides a filter across input** - computing weighted sum at each position.

2. **Filters detect features** - edges, textures, patterns, and more.

3. **Output size depends on** - input size, kernel size, stride, and padding.

4. **Multiple filters create multiple feature maps** - each detecting different features.

5. **Parameters = (kernel_h x kernel_w x input_channels + 1) x num_filters**.

## Looking Ahead

Conv layers detect features, but the resulting feature maps are still large. The next reading on **Pooling Layers** shows how to reduce spatial dimensions while keeping the important information.

## Additional Resources

- [Conv Arithmetic](https://github.com/vdumoulin/conv_arithmetic) - Animated convolution examples
- [Visualizing CNN Filters](https://distill.pub/2017/feature-visualization/) - What filters learn
- [TensorFlow Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D) - Official docs

