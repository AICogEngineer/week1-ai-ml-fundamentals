# Dense Layers Implementation

## Learning Objectives

- Understand Dense (fully connected) layers in detail
- Explain the parameters: units, activation, kernel initializer
- Connect Dense layers to MLP concepts from earlier this week
- Examine and manipulate weight matrices directly

## Why This Matters

The Dense layer is the workhorse of neural networks. Every MLP you've studied this week is built from Dense layers. Understanding exactly what happens inside a Dense layer - what parameters it has, how many weights it creates, how activations flow through it - completes the bridge between your mathematical understanding and practical TensorFlow code.

In our **From Zero to Neural** journey, this reading ensures you understand not just *how* to use Dense layers, but *why* they work the way they do.

## The Concept

### What Is a Dense Layer?

A **Dense layer** (also called fully connected layer) implements this operation:

```
output = activation(dot(input, weights) + bias)

Or in math notation:
a = activation(W @ x + b)

Where:
  x = input vector (n_in features)
  W = weight matrix (n_in x n_out)
  b = bias vector (n_out)
  a = output vector (n_out features)
```

**"Dense" means every input connects to every output:**

```
Input (3 neurons)          Output (4 neurons)

    x1 ----+----+----+-----> y1
           |    |    |
    x2 ----+----+----+-----> y2
           |    |    |
    x3 ----+----+----+-----> y3
                |    |
                +----+-----> y4

Every input connects to every output = "fully connected" = "dense"
```

### Dense Layer Parameters

```python
keras.layers.Dense(
    units,                    # Number of neurons
    activation=None,          # Activation function
    use_bias=True,            # Include bias term?
    kernel_initializer='glorot_uniform',  # Weight initialization
    bias_initializer='zeros',             # Bias initialization
    kernel_regularizer=None,  # Weight regularization
    bias_regularizer=None,    # Bias regularization
    activity_regularizer=None,# Output regularization
    kernel_constraint=None,   # Weight constraints
    bias_constraint=None,     # Bias constraints
)
```

**Key Parameters:**

| Parameter | Description | Common Values |
|-----------|-------------|---------------|
| `units` | Number of output neurons | 32, 64, 128, 256 |
| `activation` | Non-linearity to apply | 'relu', 'sigmoid', 'softmax' |
| `use_bias` | Include bias term | True (default) |
| `kernel_initializer` | How to initialize weights | 'glorot_uniform', 'he_normal' |

### Mapping Dense to MLP Concepts

**Wednesday's MLP:**
```
Layer 1: z = W @ x + b, then a = ReLU(z)
```

**Keras Dense:**
```python
layers.Dense(64, activation='relu')
# Does exactly the same thing!
```

**Correspondence:**

| MLP Concept | Dense Layer |
|-------------|-------------|
| Number of neurons | `units` |
| Weight matrix W | `kernel` (internally) |
| Bias vector b | `bias` (internally) |
| Activation function | `activation` |
| Forward pass | `layer(input)` |

### Weight Matrix Dimensions

For a Dense layer receiving `n_in` features and outputting `n_out` features:

```
Weight matrix shape: (n_in, n_out)
Bias shape: (n_out,)

Parameters = (n_in * n_out) + n_out
            = n_out * (n_in + 1)
```

**Example:**
```python
# Input: 784 features, Output: 128 neurons
layer = layers.Dense(128, input_shape=(784,))

# Weights: 784 x 128 = 100,352
# Biases: 128
# Total: 100,480 parameters
```

### Weight Initialization

How weights start affects whether the network can learn:

**Glorot/Xavier Uniform (default):**
```python
# Weights drawn from uniform distribution
# Range: [-limit, limit] where limit = sqrt(6 / (fan_in + fan_out))
layers.Dense(64, kernel_initializer='glorot_uniform')
```

**He Normal (recommended for ReLU):**
```python
# Weights drawn from normal distribution
# Std: sqrt(2 / fan_in)
layers.Dense(64, kernel_initializer='he_normal', activation='relu')
```

**Why It Matters:**
```
Bad initialization:           Good initialization:
- Activations too large       - Activations stay reasonable
- Activations too small       - Gradients flow well
- Gradients explode/vanish    - Learning proceeds smoothly
```

### Accessing Layer Weights

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Create a simple model
model = keras.Sequential([
    layers.Dense(8, activation='relu', input_shape=(4,), name='hidden'),
    layers.Dense(2, activation='softmax', name='output')
])

# Build the model (creates weights)
model.build()

# Access weights
hidden_layer = model.get_layer('hidden')
weights, biases = hidden_layer.get_weights()

print(f"Weight matrix shape: {weights.shape}")  # (4, 8)
print(f"Bias vector shape: {biases.shape}")     # (8,)
print(f"Weights:\n{weights}")
print(f"Biases: {biases}")
```

### Manual Forward Pass

Let's verify that Dense does what we expect:

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Create input
x = np.array([[1.0, 2.0, 3.0, 4.0]])  # Shape (1, 4)

# Create Dense layer
layer = layers.Dense(3, activation='relu', input_shape=(4,))

# Build layer (creates weights)
_ = layer(x)  # First call builds the layer

# Get weights
W, b = layer.get_weights()
print(f"Weight shape: {W.shape}")  # (4, 3)
print(f"Bias shape: {b.shape}")    # (3,)

# Manual computation
z = np.dot(x, W) + b  # Linear transformation
a_manual = np.maximum(0, z)  # ReLU activation

# Keras computation
a_keras = layer(x).numpy()

# Compare
print(f"\nManual result: {a_manual}")
print(f"Keras result:  {a_keras}")
print(f"Match: {np.allclose(a_manual, a_keras)}")
```

### Common Dense Layer Patterns

**Hidden Layer with ReLU:**
```python
layers.Dense(128, activation='relu')
```

**Output Layer - Binary Classification:**
```python
layers.Dense(1, activation='sigmoid')
```

**Output Layer - Multi-class Classification:**
```python
layers.Dense(num_classes, activation='softmax')
```

**Output Layer - Regression:**
```python
layers.Dense(1)  # No activation (linear)
```

**With Regularization:**
```python
from tensorflow.keras import regularizers

layers.Dense(
    64,
    activation='relu',
    kernel_regularizer=regularizers.l2(0.01)  # L2 weight penalty
)
```

## Code Example: Dense Layer Deep Dive

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

print("=" * 60)
print("DENSE LAYERS DEEP DIVE")
print("=" * 60)

# === Creating Dense Layers ===
print("\n--- Dense Layer Creation ---")

# Simple Dense layer
dense1 = layers.Dense(64, activation='relu', name='dense_relu')
print(f"Created: {dense1.name}")

# Dense with custom initialization
dense2 = layers.Dense(
    32,
    activation='relu',
    kernel_initializer='he_normal',
    bias_initializer='zeros',
    name='dense_he'
)
print(f"Created: {dense2.name} with He initialization")

# === Weight Inspection ===
print("\n--- Weight Inspection ---")

# Create model to establish weights
model = keras.Sequential([
    layers.Dense(8, activation='relu', input_shape=(4,), name='hidden'),
    layers.Dense(3, activation='softmax', name='output')
])

# Print weight details
for layer in model.layers:
    if layer.weights:
        kernel, bias = layer.get_weights()
        print(f"\n{layer.name}:")
        print(f"  Input features: {kernel.shape[0]}")
        print(f"  Output neurons: {kernel.shape[1]}")
        print(f"  Kernel shape: {kernel.shape}")
        print(f"  Bias shape: {bias.shape}")
        print(f"  Total params: {kernel.size + bias.size}")
        print(f"  Kernel mean: {kernel.mean():.6f}, std: {kernel.std():.6f}")

# === Manual vs Keras Comparison ===
print("\n--- Manual Forward Pass Verification ---")

# Fixed input
x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)

# Get hidden layer weights
hidden_layer = model.get_layer('hidden')
W, b = hidden_layer.get_weights()

# Manual computation
z_manual = np.dot(x, W) + b  # Linear
a_manual = np.maximum(0, z_manual)  # ReLU

# Keras computation
a_keras = hidden_layer(x).numpy()

print(f"Input: {x}")
print(f"Manual output: {a_manual}")
print(f"Keras output:  {a_keras}")
print(f"Outputs match: {np.allclose(a_manual, a_keras)}")

# === Setting Custom Weights ===
print("\n--- Setting Custom Weights ---")

# Create a predictable layer
custom_layer = layers.Dense(2, use_bias=False, input_shape=(3,))
_ = custom_layer(np.zeros((1, 3)))  # Build layer

# Set weights to known values
custom_weights = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
], dtype=np.float32)

custom_layer.set_weights([custom_weights])

# Test with simple input
test_input = np.array([[1.0, 2.0, 3.0]])
output = custom_layer(test_input).numpy()

print(f"Custom weights:\n{custom_weights}")
print(f"Input: {test_input}")
print(f"Output: {output}")
print(f"Expected: [1*1 + 2*0 + 3*1, 1*0 + 2*1 + 3*1] = [4, 5]")

# === Parameter Count Verification ===
print("\n--- Parameter Count Formula ---")

def count_params(input_dim, units, use_bias=True):
    """Calculate parameters for a Dense layer."""
    kernel_params = input_dim * units
    bias_params = units if use_bias else 0
    return kernel_params + bias_params

# Test against Keras
test_cases = [
    (784, 128, True),
    (128, 64, True),
    (64, 10, False),
]

print(f"{'Input':<8} {'Units':<8} {'Bias':<6} {'Formula':<12} {'Keras':<12}")
print("-" * 50)

for input_dim, units, use_bias in test_cases:
    layer = layers.Dense(units, use_bias=use_bias, input_shape=(input_dim,))
    layer.build((None, input_dim))
    
    formula_count = count_params(input_dim, units, use_bias)
    keras_count = layer.count_params()
    
    print(f"{input_dim:<8} {units:<8} {str(use_bias):<6} {formula_count:<12} {keras_count:<12}")

# === Dense Layer in Network ===
print("\n--- Dense Layers in Full Network ---")

network = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # 784 features
    layers.Dense(256, activation='relu', name='dense_1'),
    layers.Dense(128, activation='relu', name='dense_2'),
    layers.Dense(64, activation='relu', name='dense_3'),
    layers.Dense(10, activation='softmax', name='output')
])

network.summary()

# Trace shapes through network
print("\n--- Shape Flow ---")
print("Input: (batch, 28, 28)")
print("After Flatten: (batch, 784)")
print("After Dense 1: (batch, 256)")
print("After Dense 2: (batch, 128)")
print("After Dense 3: (batch, 64)")
print("Output: (batch, 10) - probabilities for 10 classes")

print("\n" + "=" * 60)
```

**Sample Output:**
```
============================================================
DENSE LAYERS DEEP DIVE
============================================================

--- Dense Layer Creation ---
Created: dense_relu
Created: dense_he with He initialization

--- Weight Inspection ---

hidden:
  Input features: 4
  Output neurons: 8
  Kernel shape: (4, 8)
  Bias shape: (8,)
  Total params: 40
  Kernel mean: -0.012345, std: 0.456789

output:
  Input features: 8
  Output neurons: 3
  Kernel shape: (8, 3)
  Bias shape: (3,)
  Total params: 27
  Kernel mean: 0.023456, std: 0.345678

--- Manual Forward Pass Verification ---
Input: [[1. 2. 3. 4.]]
Manual output: [[0.   0.   1.23 0.   0.45 0.   0.78 0.  ]]
Keras output:  [[0.   0.   1.23 0.   0.45 0.   0.78 0.  ]]
Outputs match: True

--- Setting Custom Weights ---
Custom weights:
[[1. 0.]
 [0. 1.]
 [1. 1.]]
Input: [[1. 2. 3.]]
Output: [[4. 5.]]
Expected: [1*1 + 2*0 + 3*1, 1*0 + 2*1 + 3*1] = [4, 5]

--- Parameter Count Formula ---
Input    Units    Bias   Formula      Keras       
--------------------------------------------------
784      128      True   100480       100480      
128      64       True   8256         8256        
64       10       False  640          640         

--- Dense Layers in Full Network ---
Model: "sequential"
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
flatten (Flatten)           (None, 784)               0         
dense_1 (Dense)             (None, 256)               200960    
dense_2 (Dense)             (None, 128)               32896     
dense_3 (Dense)             (None, 64)                8256      
output (Dense)              (None, 10)                650       
=================================================================
Total params: 242,762
Trainable params: 242,762
Non-trainable params: 0
_________________________________________________________________

--- Shape Flow ---
Input: (batch, 28, 28)
After Flatten: (batch, 784)
After Dense 1: (batch, 256)
After Dense 2: (batch, 128)
After Dense 3: (batch, 64)
Output: (batch, 10) - probabilities for 10 classes

============================================================
```

### Connection to MLP Concepts

| Wednesday's MLP Concept | Thursday's Keras Implementation |
|------------------------|--------------------------------|
| Neuron with weights w, bias b | `Dense(1)` |
| Layer of n neurons | `Dense(n)` |
| ReLU activation | `activation='relu'` |
| Sigmoid output | `activation='sigmoid'` |
| Forward pass z = Wx + b | `layer(x)` |
| Weight matrix W | `layer.kernel` |
| Bias vector b | `layer.bias` |

## Key Takeaways

1. **Dense = fully connected** - every input connects to every output.

2. **Parameters = (input_features x units) + units** - weights plus biases.

3. **Activation is applied after linear transformation** - same as your MLP math.

4. **Initialization matters** - use 'he_normal' for ReLU layers.

5. **You can access and modify weights** - `get_weights()`, `set_weights()`.

## Week 1 Closure

Congratulations! You've completed the Thursday readings. You now understand:
- TensorFlow as a framework
- Tensors and their shapes
- Graphs and operations
- Keras Sequential API
- Dense layers in detail

Tomorrow (Friday), you'll learn about **Convolutional Neural Networks** - a specialized architecture that revolutionized computer vision. The Dense layers you've mastered today will serve as the classification head of those networks.

## Additional Resources

- [Keras Dense Layer](https://keras.io/api/layers/core_layers/dense/) - Official API reference
- [Weight Initialization Strategies](https://www.tensorflow.org/api_docs/python/tf/keras/initializers) - All initializers
- [Understanding Dense Layers](https://machinelearningmastery.com/neural-network-in-keras/) - Tutorial with examples

