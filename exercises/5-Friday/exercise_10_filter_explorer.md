# Exercise 10: Filter Explorer Lab

## Learning Objectives

- Understand how convolution filters work
- Apply handcrafted filters to images
- Visualize feature maps
- Build intuition for what CNNs "see"

## Duration

**Estimated Time:** 60 minutes

---

## Background

Before CNNs learn their own filters, let's manually apply classic image processing filters to understand what convolution does.

---

## Part 1: Load and Display an Image (10 min)

```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load a sample image (use a built-in or your own)
# Option A: Use TensorFlow's sample image
sample = tf.keras.preprocessing.image.load_img(
    tf.keras.utils.get_file('cat.jpg', 
    'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg'),
    target_size=(224, 224)
)
image = np.array(sample)

# Convert to grayscale for simpler filtering
gray = np.mean(image, axis=2)

# TODO: Display the original and grayscale images
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(image)
axes[0].set_title('Original')
axes[1].imshow(gray, cmap='gray')
axes[1].set_title('Grayscale')
plt.show()
```

---

## Part 2: Define Classic Filters (15 min)

### Task 2.1: Edge Detection Filters

```python
# Sobel filters (edge detection)
sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

sobel_y = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=np.float32)

# TODO: Create a simple vertical edge detector
vertical_edge = np.array([
    # Your 3x3 filter
], dtype=np.float32)

# TODO: Create a simple horizontal edge detector
horizontal_edge = np.array([
    # Your 3x3 filter
], dtype=np.float32)
```

### Task 2.2: Other Filters

```python
# Sharpening filter
sharpen = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
], dtype=np.float32)

# Blur filter (box blur)
blur = np.array([
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9]
], dtype=np.float32)

# TODO: Create a Gaussian blur (3x3 approximation)
gaussian_blur = np.array([
    # Hint: Center should be highest, corners lowest
], dtype=np.float32)

# Emboss filter
emboss = np.array([
    [-2, -1, 0],
    [-1,  1, 1],
    [ 0,  1, 2]
], dtype=np.float32)
```

---

## Part 3: Apply Filters Using TensorFlow (20 min)

### Task 3.1: Create Convolution Function

```python
def apply_filter(image, kernel):
    """
    Apply a single filter to a grayscale image using TensorFlow.
    
    Args:
        image: 2D numpy array (height, width)
        kernel: 2D numpy array (kh, kw)
    
    Returns:
        Filtered image
    """
    # Reshape for TensorFlow: (batch, height, width, channels)
    img_tensor = image.reshape(1, image.shape[0], image.shape[1], 1)
    img_tensor = tf.cast(img_tensor, tf.float32)
    
    # Reshape kernel: (kh, kw, in_channels, out_channels)
    kernel_tensor = kernel.reshape(kernel.shape[0], kernel.shape[1], 1, 1)
    kernel_tensor = tf.cast(kernel_tensor, tf.float32)
    
    # Apply convolution
    output = tf.nn.conv2d(img_tensor, kernel_tensor, strides=1, padding='SAME')
    
    return output.numpy().squeeze()

# TODO: Test with sobel_x
filtered = apply_filter(gray, sobel_x)
plt.imshow(filtered, cmap='gray')
plt.title('Sobel X (Vertical Edges)')
plt.show()
```

### Task 3.2: Apply All Filters

```python
filters = {
    'Sobel X': sobel_x,
    'Sobel Y': sobel_y,
    'Sharpen': sharpen,
    'Blur': blur,
    'Emboss': emboss
}

# TODO: Create a grid showing all filter results
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

axes[0].imshow(gray, cmap='gray')
axes[0].set_title('Original')

for i, (name, kernel) in enumerate(filters.items(), start=1):
    filtered = apply_filter(gray, kernel)
    axes[i].imshow(filtered, cmap='gray')
    axes[i].set_title(name)

plt.tight_layout()
plt.show()
```

---

## Part 4: Edge Magnitude (10 min)

Combine Sobel X and Y to get edge magnitude:

```python
# Edge magnitude = sqrt(sobel_x^2 + sobel_y^2)
edges_x = apply_filter(gray, sobel_x)
edges_y = apply_filter(gray, sobel_y)

# TODO: Calculate edge magnitude
edge_magnitude = None  # np.sqrt(edges_x**2 + edges_y**2)

# TODO: Display original vs edge magnitude
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(gray, cmap='gray')
axes[0].set_title('Original')
axes[1].imshow(edge_magnitude, cmap='gray')
axes[1].set_title('Edge Magnitude')
plt.show()
```

---

## Part 5: Create Your Own Filters (Bonus)

### Challenge A: Diagonal Edge Detector

```python
# TODO: Create a filter that detects diagonal edges (/)
diagonal_filter = np.array([
    # Your filter
], dtype=np.float32)
```

### Challenge B: Corner Detector

```python
# TODO: Create filters that respond to corners
# Hint: Think about what pattern a corner makes
```

### Challenge C: Custom Pattern

```python
# TODO: Design a filter that detects a specific pattern
# (e.g., circles, horizontal lines, etc.)
```

---

## Part 6: Reflection

Answer these questions:

```python
# Q1: What do positive and negative values in a filter represent?
# Answer:

# Q2: Why does the blur filter have all positive values that sum to 1?
# Answer:

# Q3: Looking at the Sobel results, why does X detect vertical edges
#     and Y detect horizontal edges? (It seems backwards!)
# Answer:

# Q4: How do you think CNN filters differ from these handcrafted ones?
# Answer:
```

---

## Definition of Done

- [ ] Image loaded and converted to grayscale
- [ ] At least 5 classic filters defined
- [ ] Convolution function implemented
- [ ] All filter results visualized
- [ ] Edge magnitude calculated
- [ ] At least 1 custom filter created
- [ ] Reflection questions answered

