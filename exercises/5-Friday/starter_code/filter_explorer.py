"""
Exercise 10: Filter Explorer Lab
================================

Apply handcrafted filters to understand convolution.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

print("=" * 60)
print("FILTER EXPLORER LAB")
print("=" * 60)

# =============================================================================
# PART 1: LOAD IMAGE
# =============================================================================

print("\n--- LOADING IMAGE ---")

# Create a simple test image with clear edges
# (Or load your own image)
image = np.zeros((100, 100), dtype=np.float32)
image[20:80, 20:80] = 1.0  # White square on black background
image[40:60, 40:60] = 0.5  # Gray square inside

# Add some diagonal lines
for i in range(100):
    if i < 100:
        image[i, min(i, 99)] = 1.0

print(f"Image shape: {image.shape}")

# TODO: Display the image
# plt.figure(figsize=(6, 6))
# plt.imshow(image, cmap='gray')
# plt.title('Test Image')
# plt.colorbar()
# plt.show()

# =============================================================================
# PART 2: DEFINE FILTERS
# =============================================================================

print("\n--- DEFINING FILTERS ---")

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

# TODO: Create vertical edge detector
vertical_edge = np.array([
    # Your 3x3 filter
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
], dtype=np.float32)

# TODO: Create horizontal edge detector
horizontal_edge = np.array([
    # Your 3x3 filter
], dtype=np.float32)

# Sharpen
sharpen = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
], dtype=np.float32)

# Blur
blur = np.array([
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9]
], dtype=np.float32)

# Emboss
emboss = np.array([
    [-2, -1, 0],
    [-1,  1, 1],
    [ 0,  1, 2]
], dtype=np.float32)

# =============================================================================
# PART 3: APPLY FILTERS
# =============================================================================

print("\n--- APPLYING FILTERS ---")

def apply_filter(image, kernel):
    """Apply a filter using TensorFlow convolution."""
    # Reshape for TensorFlow
    img_tensor = image.reshape(1, image.shape[0], image.shape[1], 1)
    img_tensor = tf.cast(img_tensor, tf.float32)
    
    kernel_tensor = kernel.reshape(kernel.shape[0], kernel.shape[1], 1, 1)
    kernel_tensor = tf.cast(kernel_tensor, tf.float32)
    
    # Apply convolution
    output = tf.nn.conv2d(img_tensor, kernel_tensor, strides=1, padding='SAME')
    
    return output.numpy().squeeze()


# TODO: Apply all filters and visualize
filters = {
    'Sobel X': sobel_x,
    'Sobel Y': sobel_y,
    'Sharpen': sharpen,
    'Blur': blur,
    'Emboss': emboss
}

# fig, axes = plt.subplots(2, 3, figsize=(15, 10))
# axes = axes.flatten()
# 
# axes[0].imshow(image, cmap='gray')
# axes[0].set_title('Original')
# 
# for i, (name, kernel) in enumerate(filters.items(), start=1):
#     filtered = apply_filter(image, kernel)
#     axes[i].imshow(filtered, cmap='gray')
#     axes[i].set_title(name)
# 
# plt.tight_layout()
# plt.show()

# =============================================================================
# PART 4: EDGE MAGNITUDE
# =============================================================================

print("\n--- EDGE MAGNITUDE ---")

# TODO: Combine Sobel X and Y
# edges_x = apply_filter(image, sobel_x)
# edges_y = apply_filter(image, sobel_y)
# edge_magnitude = np.sqrt(edges_x**2 + edges_y**2)

# =============================================================================
# PART 5: CUSTOM FILTERS
# =============================================================================

print("\n--- CUSTOM FILTERS ---")

# TODO: Create diagonal edge detector
diagonal_filter = np.array([
    # Your filter
], dtype=np.float32)

# =============================================================================
# REFLECTION QUESTIONS
# =============================================================================

# Q1: What do positive and negative values in a filter represent?
# Answer:

# Q2: Why does blur have all positive values summing to 1?
# Answer:

# Q3: Why does Sobel X detect vertical edges?
# Answer:

# Q4: How do CNN filters differ from these handcrafted ones?
# Answer:

