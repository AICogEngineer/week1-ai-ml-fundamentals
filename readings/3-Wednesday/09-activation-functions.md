# Activation Functions

## Learning Objectives

- Explain why activation functions are essential (introducing non-linearity)
- Describe the properties and behavior of Step, Sigmoid, Tanh, ReLU, and Leaky ReLU
- Interpret activation function graphs and understand their outputs
- Select appropriate activation functions for different network positions

## Why This Matters

Without activation functions, a neural network - no matter how many layers - would be nothing more than a fancy linear regression. The activation function is what transforms neural networks from expensive matrix multipliers into universal function approximators.

In our **From Zero to Neural** journey, activation functions are the key ingredient that unlocks non-linear learning. They're why neural networks can recognize faces, understand language, and solve problems that no linear model ever could.

## The Concept

### Why Non-Linearity Matters

Consider a network with no activation functions (or only linear activations):

```
Layer 1: z1 = W1 * x + b1
Layer 2: z2 = W2 * z1 + b2
Layer 3: z3 = W3 * z2 + b3

Substituting:
z3 = W3 * (W2 * (W1 * x + b1) + b2) + b3
z3 = (W3 * W2 * W1) * x + (W3 * W2 * b1 + W3 * b2 + b3)
z3 = W_combined * x + b_combined

Three layers collapse to one linear transformation!
```

**Adding non-linear activations breaks this collapse**, allowing each layer to learn distinct representations.

### Activation Function Requirements

Good activation functions should:
1. **Be non-linear** - to enable learning complex patterns
2. **Be differentiable** - to allow gradient-based training
3. **Have bounded outputs** (sometimes) - to prevent exploding values
4. **Avoid "dead" regions** - where gradients are zero

### 1. Step Function (Historical)

The original perceptron activation - outputs binary 0 or 1.

```
        Output
          |
        1 |      ___________
          |     |
          |     |
        0 |_____|
          +------|---------> z
                 0

f(z) = 1  if z >= 0
f(z) = 0  if z < 0
```

**Properties:**
- Non-differentiable at z=0 (derivative is 0 everywhere else)
- Binary output
- Cannot use gradient descent for training

```python
def step(z):
    return np.where(z >= 0, 1, 0)
```

**When used**: Original perceptrons (historical), final layer of binary classification with thresholding

### 2. Sigmoid (Logistic)

Smooths the step function into an S-curve, outputting values between 0 and 1.

```
        Output
        1 |           *****
          |         **
      0.5 |--------*---------
          |      **
        0 |*****
          +------|---------> z
               -4  0  4

f(z) = 1 / (1 + e^(-z))
```

**Properties:**
- Output range: (0, 1) - interpretable as probability
- Smooth, differentiable everywhere
- Derivative: f'(z) = f(z) * (1 - f(z))
- Saturates at extremes (gradient nearly 0)

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)
```

**Problems:**
- **Vanishing gradient**: For large |z|, gradient approaches 0, stalling learning
- **Not zero-centered**: Outputs always positive, causing zig-zag gradient updates
- **Computationally expensive**: exp() is slow

**When used**: Output layer for binary classification (probability output)

### 3. Tanh (Hyperbolic Tangent)

Like sigmoid but zero-centered, outputting values between -1 and 1.

```
        Output
        1 |           *****
          |         **
        0 |--------*---------
          |      **
       -1 |*****
          +------|---------> z
               -4  0  4

f(z) = (e^z - e^(-z)) / (e^z + e^(-z))
     = 2 * sigmoid(2z) - 1
```

**Properties:**
- Output range: (-1, 1)
- Zero-centered (fixes sigmoid's zig-zag problem)
- Derivative: f'(z) = 1 - f(z)^2
- Still saturates at extremes

```python
def tanh(z):
    return np.tanh(z)

def tanh_derivative(z):
    return 1 - np.tanh(z)**2
```

**When used**: Hidden layers (better than sigmoid), RNNs, sometimes output layer for [-1, 1] range

### 4. ReLU (Rectified Linear Unit)

The breakthrough activation that enabled deep learning.

```
        Output
          |       /
          |      /
          |     /
          |    /
        0 |___/
          +------|---------> z
                 0

f(z) = max(0, z)
```

**Properties:**
- Output range: [0, infinity)
- Computationally cheap: just max(0, z)
- Derivative: 1 if z > 0, else 0
- No saturation for positive values
- Sparse activation (many neurons output 0)

```python
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return np.where(z > 0, 1, 0)
```

**Problems:**
- **Dying ReLU**: Neurons that output 0 can get stuck (gradient is 0)
- **Not zero-centered**: Outputs always >= 0

**When used**: Default for hidden layers in most modern networks

### 5. Leaky ReLU

Fixes the dying ReLU problem by allowing small negative outputs.

```
        Output
          |       /
          |      /
          |     /
        0 |---_/
          | _/
          |/
          +------|---------> z
                 0

f(z) = max(alpha * z, z)
     = z if z > 0
     = alpha * z if z <= 0

(typically alpha = 0.01)
```

**Properties:**
- Output range: (-infinity, infinity)
- Small gradient for negative values (prevents dying)
- Derivative: 1 if z > 0, else alpha

```python
def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def leaky_relu_derivative(z, alpha=0.01):
    return np.where(z > 0, 1, alpha)
```

**Variants:**
- **Parametric ReLU (PReLU)**: alpha is learned during training
- **ELU**: Uses exponential for negative values, smoother

**When used**: Hidden layers when dying ReLU is a concern

### Comparison Summary

| Function | Output Range | Zero-Centered | Gradient Issues | Speed |
|----------|--------------|---------------|-----------------|-------|
| Step | {0, 1} | No | Non-differentiable | Fast |
| Sigmoid | (0, 1) | No | Vanishing | Slow |
| Tanh | (-1, 1) | Yes | Vanishing | Slow |
| ReLU | [0, inf) | No | Dying neurons | Very Fast |
| Leaky ReLU | (-inf, inf) | No | Minimal | Very Fast |

### Visual Comparison

```
Sigmoid:              Tanh:                 ReLU:               Leaky ReLU:

1|    ___             1|    ___             |      /             |      /
 |   /                 |   /                |     /              |     /
 |  /                 0|--*--               |    /              0|---_/
 | /                   |   \               0|___/                | _/
0|___                -1|___                  |                    |

(smooth S-curve)     (zero-centered)      (simple, sparse)    (no dead neurons)
```

## Code Example: Comparing Activations

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def tanh_activation(z):
    return np.tanh(z)

def relu(z):
    return np.maximum(0, z)

def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

# Create input range
z = np.linspace(-5, 5, 100)

# Calculate activations
activations = {
    'Sigmoid': sigmoid(z),
    'Tanh': tanh_activation(z),
    'ReLU': relu(z),
    'Leaky ReLU': leaky_relu(z)
}

# Display values at key points
print("Activation values at key inputs:")
print(f"{'Input':<10} {'Sigmoid':<12} {'Tanh':<12} {'ReLU':<12} {'Leaky ReLU':<12}")
print("-" * 58)
for val in [-2, -1, 0, 1, 2]:
    print(f"{val:<10} {sigmoid(val):<12.4f} {tanh_activation(val):<12.4f} {relu(val):<12.4f} {leaky_relu(val):<12.4f}")
```

**Output:**
```
Activation values at key inputs:
Input      Sigmoid      Tanh         ReLU         Leaky ReLU  
----------------------------------------------------------
-2         0.1192       -0.9640      0.0000       -0.0200     
-1         0.2689       -0.7616      0.0000       -0.0100     
0          0.5000       0.0000       0.0000       0.0000      
1          0.7311       0.7616       1.0000       1.0000      
2          0.8808       0.9640       2.0000       2.0000      
```

### Derivatives (Gradients) Comparison

```python
def sigmoid_grad(z):
    s = sigmoid(z)
    return s * (1 - s)

def tanh_grad(z):
    return 1 - np.tanh(z)**2

def relu_grad(z):
    return np.where(z > 0, 1, 0)

def leaky_relu_grad(z, alpha=0.01):
    return np.where(z > 0, 1, alpha)

print("\nGradient values at key inputs:")
print(f"{'Input':<10} {'Sigmoid':<12} {'Tanh':<12} {'ReLU':<12} {'Leaky ReLU':<12}")
print("-" * 58)
for val in [-2, -1, 0, 1, 2]:
    print(f"{val:<10} {sigmoid_grad(val):<12.4f} {tanh_grad(val):<12.4f} {relu_grad(val):<12.4f} {leaky_relu_grad(val):<12.4f}")
```

**Output:**
```
Gradient values at key inputs:
Input      Sigmoid      Tanh         ReLU         Leaky ReLU  
----------------------------------------------------------
-2         0.1050       0.0707       0.0000       0.0100      
-1         0.1966       0.4200       0.0000       0.0100      
0          0.2500       1.0000       0.0000       0.0100      
1          0.1966       0.4200       1.0000       1.0000      
2          0.1050       0.0707       1.0000       1.0000      
```

Notice:
- Sigmoid/Tanh gradients are small at extremes (vanishing gradient)
- ReLU gradient is 0 for negative inputs (dying ReLU)
- Leaky ReLU maintains small gradient for negative inputs

### Choosing Activation Functions

**Rules of Thumb:**

```
Hidden Layers:
  - Default choice: ReLU (fast, works well)
  - If dying ReLU is an issue: Leaky ReLU or ELU
  - For RNNs: Tanh (bounded outputs help with sequences)

Output Layer (depends on task):
  - Binary classification: Sigmoid (outputs probability 0-1)
  - Multi-class classification: Softmax (outputs probability distribution)
  - Regression: Linear (no activation) or ReLU for positive outputs
  - Bounded regression: Sigmoid or Tanh
```

**Network Architecture Pattern:**

```
Input -> [ReLU] -> [ReLU] -> [ReLU] -> [Sigmoid] -> Output
         Hidden    Hidden    Hidden    Output
         Layers    Layers    Layers    Layer
```

## Key Takeaways

1. **Activation functions introduce non-linearity** - without them, deep networks collapse to linear models.

2. **Sigmoid outputs probabilities (0-1)** but suffers from vanishing gradients; use for binary classification output.

3. **Tanh is zero-centered (-1 to 1)** which improves gradient flow; good for RNNs and some hidden layers.

4. **ReLU is the modern default** - simple, fast, enables deep learning; watch for dying neurons.

5. **Leaky ReLU fixes dying neurons** by allowing small negative gradients.

## Looking Ahead

Now that you understand individual neurons and their activation functions, the next reading on **Multi-Layer Perceptrons** shows how to stack these neurons into powerful networks that can learn any function - solving problems like XOR that single perceptrons cannot.

## Additional Resources

- [Activation Functions - Stanford CS231n](https://cs231n.github.io/neural-networks-1/#actfun) - Deep dive with practical advice
- [ReLU and Its Variants](https://paperswithcode.com/methods/category/activation-functions) - Modern activation functions
- [Visualizing Activations](https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/) - When to use what

