# Exercise 05: Activation Function Analysis

## Learning Objectives

- Implement common activation functions
- Understand their properties through experimentation
- Analyze gradients and their implications for learning
- Document observations systematically

## Duration

**Estimated Time:** 60 minutes

## Type

**Hybrid Lab:** Part implementation, part analysis and documentation

---

## Part 1: Implement Activations (20 min)

### Task 1.1: Code the Functions

Navigate to `starter_code/activation_analysis.py`:

```python
import numpy as np
import matplotlib.pyplot as plt

def step(z):
    """Step function (Heaviside)."""
    # TODO: Return 1 if z >= 0, else 0
    pass

def sigmoid(z):
    """Sigmoid / Logistic function."""
    # TODO: Return 1 / (1 + exp(-z))
    # Hint: Use np.clip to avoid overflow
    pass

def tanh_activation(z):
    """Hyperbolic tangent."""
    # TODO: Return tanh(z)
    pass

def relu(z):
    """Rectified Linear Unit."""
    # TODO: Return max(0, z)
    pass

def leaky_relu(z, alpha=0.01):
    """Leaky ReLU."""
    # TODO: Return z if z > 0, else alpha * z
    pass
```

### Task 1.2: Implement Derivatives

```python
def sigmoid_derivative(z):
    """Derivative of sigmoid: sigmoid(z) * (1 - sigmoid(z))"""
    # TODO: Implement
    pass

def tanh_derivative(z):
    """Derivative of tanh: 1 - tanh(z)^2"""
    # TODO: Implement
    pass

def relu_derivative(z):
    """Derivative of ReLU: 1 if z > 0, else 0"""
    # TODO: Implement
    pass

def leaky_relu_derivative(z, alpha=0.01):
    """Derivative of Leaky ReLU: 1 if z > 0, else alpha"""
    # TODO: Implement
    pass
```

---

## Part 2: Visualization (15 min)

### Task 2.1: Plot All Activations

```python
z = np.linspace(-5, 5, 1000)

# TODO: Create a 2x3 subplot grid
# Plot each activation function
# Include the function equation in the title
```

### Task 2.2: Plot Derivatives

```python
# TODO: Create another 2x3 subplot for derivatives
# This shows how fast each function changes
```

---

## Part 3: Analysis Experiments (25 min)

### Experiment A: Output Ranges

Fill in the table:

| Activation | Min Output | Max Output | Zero-Centered? |
|------------|------------|------------|----------------|
| Step       | ?          | ?          | ?              |
| Sigmoid    | ?          | ?          | ?              |
| Tanh       | ?          | ?          | ?              |
| ReLU       | ?          | ?          | ?              |
| Leaky ReLU | ?          | ?          | ?              |

```python
# TODO: Verify by computing actual min/max values
# for activation in [step, sigmoid, tanh_activation, relu, leaky_relu]:
#     outputs = activation(z)
#     print(f"{activation.__name__}: min={outputs.min():.4f}, max={outputs.max():.4f}")
```

### Experiment B: Gradient Values

Examine gradient magnitudes:

```python
# TODO: For z = -3, -1, 0, 1, 3
# Calculate and print gradient values for each activation
# 
# test_points = [-3, -1, 0, 1, 3]
# for z_val in test_points:
#     print(f"z = {z_val}")
#     print(f"  Sigmoid gradient: {sigmoid_derivative(z_val):.6f}")
#     print(f"  Tanh gradient: {tanh_derivative(z_val):.6f}")
#     print(f"  ReLU gradient: {relu_derivative(z_val):.6f}")
```

### Experiment C: The Vanishing Gradient Problem

```python
# TODO: What happens to sigmoid gradient when z = 10? z = -10?
# Calculate and explain why this is a problem for deep networks.
#
# z_extreme = 10
# print(f"Sigmoid({z_extreme}) = {sigmoid(z_extreme):.10f}")
# print(f"Sigmoid gradient at z={z_extreme}: {sigmoid_derivative(z_extreme):.10f}")
```

### Experiment D: Dead ReLU

```python
# TODO: What is ReLU's gradient when z < 0?
# Explain the "dying ReLU" problem.
# How does Leaky ReLU address this?
```

---

## Part 4: Written Analysis

Complete the analysis template in `templates/activation_analysis.md`:

```markdown
# Activation Function Analysis Report

## 1. Output Range Comparison

| Activation | Range | Best For |
|------------|-------|----------|
| Sigmoid    |       |          |
| Tanh       |       |          |
| ReLU       |       |          |

## 2. Gradient Behavior

### Sigmoid
- At z=0: gradient = ___
- At z=5: gradient = ___
- Problem: ___

### ReLU
- At z > 0: gradient = ___
- At z < 0: gradient = ___
- Problem: ___

## 3. Recommendations

For hidden layers, I would use ___ because:
1. 
2.

For output layer (binary classification), I would use ___ because:
1.

## 4. Key Insight

The most important thing I learned about activation functions is:
```

---

## Definition of Done

- [ ] All 5 activation functions implemented
- [ ] All 4 derivatives implemented
- [ ] Visualization plots created
- [ ] Output range table completed
- [ ] Gradient experiments run
- [ ] Analysis report filled out
- [ ] Vanishing gradient explained
- [ ] Dead ReLU explained

---

## Key Takeaways

After completing this exercise, you should understand:

1. **Sigmoid**: Good for probabilities, but vanishes at extremes
2. **Tanh**: Zero-centered, but still vanishes
3. **ReLU**: Fast, no vanishing gradient for z > 0, but can "die"
4. **Leaky ReLU**: Fixes dead ReLU problem
5. **Choice matters**: Hidden layers vs output layers need different activations

