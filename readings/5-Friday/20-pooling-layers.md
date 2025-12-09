# Pooling Layers

## Learning Objectives

- Understand pooling as dimensionality reduction for feature maps
- Compare Max Pooling vs Average Pooling
- Explain how pooling provides spatial invariance
- Apply pool size and stride parameters correctly

## Why This Matters

Convolutional layers detect features, but they preserve spatial dimensions (with `padding='same'`) or reduce them only slightly. For a 224x224 image with 64 filters, you'd have 224x224x64 = 3.2 million values after the first conv layer!

Pooling aggressively reduces spatial dimensions while keeping the most important information. It's how CNNs stay computationally tractable while processing high-resolution images.

In our **From Zero to Neural** journey, pooling completes your understanding of the CNN pipeline: convolution extracts features, pooling compresses them.

## The Concept

### What Is Pooling?

**Pooling** reduces the spatial dimensions of feature maps by summarizing regions into single values.

```
Input Feature Map (4x4):        Max Pooling (2x2):      Output (2x2):

[1  3  2  4]                    Take maximum           [4  5]
[5  2  1  0]    ------>         from each 2x2         [8  6]
[2  8  3  1]                    region
[4  1  6  2]
```

### Max Pooling

**Max Pooling** takes the maximum value in each region:

```
Max Pooling 2x2 with stride 2:

Input:                          Operation:              Output:
[1  3 | 2  4]                   max(1,3,5,2) = 5       [5  4]
[5  2 | 1  0]                   max(2,4,1,0) = 4       [8  6]
[- - - - - -]                   max(2,8,4,1) = 8
[2  8 | 3  1]                   max(3,1,6,2) = 6
[4  1 | 6  2]
```

**Why Maximum?**
- Preserves the strongest activation (detected feature)
- If an edge exists anywhere in the region, max pooling keeps it
- Provides translation invariance within the pooling region

### Average Pooling

**Average Pooling** takes the mean value in each region:

```
Average Pooling 2x2 with stride 2:

Input:                          Operation:              Output:
[1  3 | 2  4]                   avg(1,3,5,2) = 2.75    [2.75  1.75]
[5  2 | 1  0]                   avg(2,4,1,0) = 1.75    [3.75  3.00]
[- - - - - -]                   avg(2,8,4,1) = 3.75
[2  8 | 3  1]                   avg(3,1,6,2) = 3.00
[4  1 | 6  2]
```

**When to Use Average Pooling:**
- When you want to preserve overall intensity information
- Global Average Pooling at the end of networks (replaces Dense layers)
- Smooth feature maps rather than sparse ones

### Max vs Average Pooling

| Aspect | Max Pooling | Average Pooling |
|--------|-------------|-----------------|
| **Operation** | Takes maximum | Takes mean |
| **Best for** | Detecting presence of features | Preserving overall intensity |
| **Output** | Sparse, strong activations | Smooth, averaged activations |
| **Common use** | Throughout CNN | End of network (Global) |
| **Translation invariance** | Strong | Moderate |

### Pooling Parameters

**1. Pool Size**
```python
# Pool size determines region size
MaxPooling2D(pool_size=(2, 2))  # 2x2 regions
MaxPooling2D(pool_size=(3, 3))  # 3x3 regions

# 2x2 is most common
```

**2. Stride**
```python
# Stride determines how far pool window moves
MaxPooling2D(pool_size=(2, 2), strides=2)  # Non-overlapping (default)
MaxPooling2D(pool_size=(3, 3), strides=2)  # Overlapping

# Default stride = pool_size (non-overlapping)
```

**3. Padding**
```python
MaxPooling2D(pool_size=(2, 2), padding='valid')  # No padding (default)
MaxPooling2D(pool_size=(2, 2), padding='same')   # Pad to preserve dimensions
```

### Output Dimension Formula

Same as convolution:
```
Output Size = floor((Input - Pool Size) / Stride) + 1

With padding='same':
Output Size = ceil(Input / Stride)
```

**Common case (2x2 pool, stride 2):**
```
Input: 28x28
Output: (28 - 2) / 2 + 1 = 14x14

Pooling halves the spatial dimensions!
```

### Why Pooling Provides Invariance

Pooling makes the network less sensitive to exact feature positions:

```
Original:              Shifted 1 pixel:
[0  0  1  0]          [0  1  0  0]
[0  0  1  0]          [0  1  0  0]
[0  0  0  0]          [0  0  0  0]
[0  0  0  0]          [0  0  0  0]

After 2x2 Max Pooling:
[0  1]                [1  0]
[0  0]                [0  0]

The "1" moved, but after pooling,
it's still in the top-left quadrant!
```

### Global Pooling

**Global Average Pooling** takes the average of the entire feature map:

```
Input: (batch, 7, 7, 512)
After GlobalAveragePooling2D:
Output: (batch, 512)

Each of 512 filters reduced to single value.
```

**Why Global Average Pooling?**
- Replaces flatten + dense layers in modern architectures
- Fewer parameters (no weights!)
- Encourages feature maps to directly represent class presence

```python
# Traditional approach:
model.add(Flatten())
model.add(Dense(1024))
model.add(Dense(num_classes))

# Modern approach (fewer params):
model.add(GlobalAveragePooling2D())
model.add(Dense(num_classes))
```

## Code Example: Pooling Operations

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

print("=" * 60)
print("POOLING LAYERS")
print("=" * 60)

# === Manual Max Pooling ===
print("\n--- Manual Max Pooling ---")

input_feature = np.array([
    [1, 3, 2, 4],
    [5, 2, 1, 0],
    [2, 8, 3, 1],
    [4, 1, 6, 2]
], dtype=np.float32)

print("Input (4x4):")
print(input_feature)

def manual_maxpool2d(x, pool_size=2, stride=2):
    h, w = x.shape
    out_h = (h - pool_size) // stride + 1
    out_w = (w - pool_size) // stride + 1
    output = np.zeros((out_h, out_w))
    
    for i in range(out_h):
        for j in range(out_w):
            region = x[i*stride:i*stride+pool_size, 
                       j*stride:j*stride+pool_size]
            output[i, j] = np.max(region)
    
    return output

def manual_avgpool2d(x, pool_size=2, stride=2):
    h, w = x.shape
    out_h = (h - pool_size) // stride + 1
    out_w = (w - pool_size) // stride + 1
    output = np.zeros((out_h, out_w))
    
    for i in range(out_h):
        for j in range(out_w):
            region = x[i*stride:i*stride+pool_size, 
                       j*stride:j*stride+pool_size]
            output[i, j] = np.mean(region)
    
    return output

maxpool_result = manual_maxpool2d(input_feature)
avgpool_result = manual_avgpool2d(input_feature)

print("\nMax Pooling 2x2 (stride 2):")
print(maxpool_result)

print("\nAverage Pooling 2x2 (stride 2):")
print(avgpool_result)

# === Keras Pooling ===
print("\n--- Keras Pooling Layers ---")

# Reshape for Keras
input_tensor = input_feature.reshape(1, 4, 4, 1)

# Max Pooling
maxpool_layer = layers.MaxPooling2D(pool_size=(2, 2), strides=2)
maxpool_output = maxpool_layer(input_tensor)
print("Keras MaxPooling2D:")
print(maxpool_output.numpy().reshape(2, 2))

# Average Pooling
avgpool_layer = layers.AveragePooling2D(pool_size=(2, 2), strides=2)
avgpool_output = avgpool_layer(input_tensor)
print("\nKeras AveragePooling2D:")
print(avgpool_output.numpy().reshape(2, 2))

# === Pooling Reduces Dimensions ===
print("\n--- Dimension Reduction ---")

# Large feature map
large_input = tf.random.normal([1, 28, 28, 32])
print(f"Input shape: {large_input.shape}")

# After pooling
pooled = layers.MaxPooling2D(2, 2)(large_input)
print(f"After MaxPool 2x2: {pooled.shape}")

pooled = layers.MaxPooling2D(2, 2)(pooled)
print(f"After 2nd MaxPool 2x2: {pooled.shape}")

# === Global Pooling ===
print("\n--- Global Pooling ---")

feature_maps = tf.random.normal([1, 7, 7, 512])
print(f"Feature maps shape: {feature_maps.shape}")

# Global Average Pooling
gap_layer = layers.GlobalAveragePooling2D()
gap_output = gap_layer(feature_maps)
print(f"After GlobalAveragePooling2D: {gap_output.shape}")

# Global Max Pooling
gmp_layer = layers.GlobalMaxPooling2D()
gmp_output = gmp_layer(feature_maps)
print(f"After GlobalMaxPooling2D: {gmp_output.shape}")

# === Pooling in CNN Architecture ===
print("\n--- Pooling in CNN Architecture ---")

model = keras.Sequential([
    # Conv block 1
    layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(2, 2),
    
    # Conv block 2
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    # Conv block 3
    layers.Conv2D(64, 3, activation='relu'),
    
    # Classification head
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.summary()

# Shape progression
print("\n--- Shape Progression ---")
x = tf.random.normal([1, 28, 28, 1])
print(f"Input: {x.shape}")

for layer in model.layers:
    x = layer(x)
    if 'pool' in layer.name or 'conv' in layer.name or 'flatten' in layer.name:
        print(f"After {layer.name}: {x.shape}")

# === No Parameters! ===
print("\n--- Pooling Has No Parameters ---")

pool_layer = layers.MaxPooling2D(2, 2)
pool_layer.build((None, 28, 28, 1))
print(f"MaxPooling2D parameters: {pool_layer.count_params()}")

print("\nPooling simply aggregates - no learning needed!")

print("\n" + "=" * 60)
```

**Sample Output:**
```
============================================================
POOLING LAYERS
============================================================

--- Manual Max Pooling ---
Input (4x4):
[[1. 3. 2. 4.]
 [5. 2. 1. 0.]
 [2. 8. 3. 1.]
 [4. 1. 6. 2.]]

Max Pooling 2x2 (stride 2):
[[5. 4.]
 [8. 6.]]

Average Pooling 2x2 (stride 2):
[[2.75 1.75]
 [3.75 3.  ]]

--- Keras Pooling Layers ---
Keras MaxPooling2D:
[[5. 4.]
 [8. 6.]]

Keras AveragePooling2D:
[[2.75 1.75]
 [3.75 3.  ]]

--- Dimension Reduction ---
Input shape: (1, 28, 28, 32)
After MaxPool 2x2: (1, 14, 14, 32)
After 2nd MaxPool 2x2: (1, 7, 7, 32)

--- Global Pooling ---
Feature maps shape: (1, 7, 7, 512)
After GlobalAveragePooling2D: (1, 512)
After GlobalMaxPooling2D: (1, 512)

--- Pooling in CNN Architecture ---
Model: "sequential"
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
conv2d (Conv2D)             (None, 26, 26, 32)        320       
max_pooling2d (MaxPooling2D)(None, 13, 13, 32)        0         
conv2d_1 (Conv2D)           (None, 11, 11, 64)        18496     
max_pooling2d_1 (MaxPooling)(None, 5, 5, 64)          0         
conv2d_2 (Conv2D)           (None, 3, 3, 64)          36928     
flatten (Flatten)           (None, 576)               0         
dense (Dense)               (None, 64)                36928     
dense_1 (Dense)             (None, 10)                650       
=================================================================
Total params: 93,322

--- Shape Progression ---
Input: (1, 28, 28, 1)
After conv2d: (1, 26, 26, 32)
After max_pooling2d: (1, 13, 13, 32)
After conv2d_1: (1, 11, 11, 64)
After max_pooling2d_1: (1, 5, 5, 64)
After conv2d_2: (1, 3, 3, 64)
After flatten: (1, 576)

--- Pooling Has No Parameters ---
MaxPooling2D parameters: 0

Pooling simply aggregates - no learning needed!

============================================================
```

## Key Takeaways

1. **Pooling reduces spatial dimensions** - typically by half with 2x2 pool and stride 2.

2. **Max pooling preserves strongest activations** - keeps detected features.

3. **Average pooling preserves overall intensity** - smoother output.

4. **Pooling has no learnable parameters** - it's a fixed operation.

5. **Global pooling converts feature maps to vectors** - useful before final classification.

## Looking Ahead

With features extracted (Conv) and compressed (Pooling), the next reading on **Flattening** explains how to connect the 2D feature maps to Dense layers for classification.

## Additional Resources

- [Pooling Layers Explained](https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/) - Detailed tutorial
- [Max Pooling vs Average Pooling](https://www.geeksforgeeks.org/cnn-introduction-to-pooling-layer/) - Comparison
- [Global Average Pooling](https://paperswithcode.com/method/global-average-pooling) - Modern usage

