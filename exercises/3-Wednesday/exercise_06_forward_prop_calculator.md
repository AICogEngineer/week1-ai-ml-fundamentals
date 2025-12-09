# Exercise 06: Forward Propagation Calculator

## Learning Objectives

- Calculate forward propagation by hand
- Verify manual calculations with NumPy code
- Understand how data transforms through network layers
- Build intuition for neural network mechanics

## Duration

**Estimated Time:** 60 minutes

## Type

**Paper + Code Exercise:** Work through calculations by hand, then verify with code

---

## The Network

You'll trace forward propagation through this network:

```
INPUT LAYER (3 features)
    |
    v
HIDDEN LAYER (2 neurons, ReLU activation)
    |
    v
OUTPUT LAYER (1 neuron, Sigmoid activation)
```

### Given Weights and Biases

**Hidden Layer:**
```
W1 = [[0.2, -0.5],    # Shape: (3, 2)
      [0.3,  0.4],
      [-0.1, 0.2]]

b1 = [0.1, -0.1]      # Shape: (2,)
```

**Output Layer:**
```
W2 = [[0.6],          # Shape: (2, 1)
      [-0.3]]

b2 = [0.1]            # Shape: (1,)
```

### Input

```
X = [1.0, 0.5, -0.5]  # Shape: (3,)
```

---

## Part 1: Paper Calculations (25 min)

Work these out by hand. Show your work!

### Step 1: Hidden Layer Linear Transform

Calculate z1 = X @ W1 + b1

```
z1[0] = X[0]*W1[0,0] + X[1]*W1[1,0] + X[2]*W1[2,0] + b1[0]
z1[0] = 1.0*0.2 + 0.5*0.3 + (-0.5)*(-0.1) + 0.1
z1[0] = _____ + _____ + _____ + 0.1
z1[0] = _____

z1[1] = X[0]*W1[0,1] + X[1]*W1[1,1] + X[2]*W1[2,1] + b1[1]
z1[1] = 1.0*(-0.5) + 0.5*0.4 + (-0.5)*0.2 + (-0.1)
z1[1] = _____ + _____ + _____ + (-0.1)
z1[1] = _____
```

**Your answer:** z1 = [_____, _____]

### Step 2: Hidden Layer Activation (ReLU)

Apply ReLU: a1 = max(0, z1)

```
a1[0] = max(0, z1[0]) = max(0, _____) = _____
a1[1] = max(0, z1[1]) = max(0, _____) = _____
```

**Your answer:** a1 = [_____, _____]

### Step 3: Output Layer Linear Transform

Calculate z2 = a1 @ W2 + b2

```
z2[0] = a1[0]*W2[0,0] + a1[1]*W2[1,0] + b2[0]
z2[0] = _____*0.6 + _____*(-0.3) + 0.1
z2[0] = _____ + _____ + 0.1
z2[0] = _____
```

**Your answer:** z2 = [_____]

### Step 4: Output Layer Activation (Sigmoid)

Apply Sigmoid: y_hat = 1 / (1 + exp(-z2))

```
y_hat = 1 / (1 + exp(-_____))
y_hat = 1 / (1 + _____)
y_hat = 1 / _____
y_hat = _____
```

**Your answer:** y_hat = _____

### Step 5: Interpret the Output

```
The network predicts: _____ (probability of positive class)

If threshold = 0.5:
Classification: _____ (0 or 1)
```

---

## Part 2: Verify with NumPy (20 min)

Navigate to `starter_code/forward_prop_calculator.py`:

```python
import numpy as np

# Given weights and biases
W1 = np.array([[0.2, -0.5],
               [0.3,  0.4],
               [-0.1, 0.2]])
b1 = np.array([0.1, -0.1])

W2 = np.array([[0.6],
               [-0.3]])
b2 = np.array([0.1])

# Input
X = np.array([1.0, 0.5, -0.5])

# Activation functions
def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# TODO: Implement forward propagation
# Step 1: z1 = ...
# Step 2: a1 = ...
# Step 3: z2 = ...
# Step 4: y_hat = ...

# TODO: Print intermediate values and compare with your hand calculations
print(f"z1 = {z1}")
print(f"a1 = {a1}")
print(f"z2 = {z2}")
print(f"y_hat = {y_hat}")
```

### Verification Checklist

- [ ] z1 matches your hand calculation
- [ ] a1 matches your hand calculation
- [ ] z2 matches your hand calculation
- [ ] y_hat matches your hand calculation

---

## Part 3: Different Inputs (15 min)

Run forward propagation for these inputs and record predictions:

### Input 2: X = [0.0, 1.0, 1.0]

```
z1 = [_____, _____]
a1 = [_____, _____]
z2 = [_____]
y_hat = _____
```

### Input 3: X = [-1.0, 0.0, 0.0]

```
z1 = [_____, _____]
a1 = [_____, _____]
z2 = [_____]
y_hat = _____
```

### Input 4: X = [2.0, 2.0, 2.0]

```
z1 = [_____, _____]
a1 = [_____, _____]
z2 = [_____]
y_hat = _____
```

### Analysis

```python
# Q1: Which input(s) produced y_hat > 0.5 (positive class)?
# Answer:

# Q2: For Input 3, what happened at the ReLU layer? Why?
# Answer:

# Q3: How does the network's output change with different inputs?
# Answer:
```

---

## Part 4: Build Forward Pass Function (Bonus)

```python
def forward_pass(X, weights, biases):
    """
    General forward pass through any network.
    
    Args:
        X: Input features
        weights: List of weight matrices [W1, W2, ...]
        biases: List of bias vectors [b1, b2, ...]
    
    Returns:
        output: Final prediction
        cache: Intermediate values for backprop
    """
    # TODO: Implement for arbitrary depth networks
    pass
```

---

## Definition of Done

- [ ] All 4 steps calculated by hand
- [ ] NumPy code verifies hand calculations
- [ ] 3 additional inputs processed
- [ ] Analysis questions answered
- [ ] Understanding demonstrated through correct answers

---

## Answer Key (Check After Completing)

<details>
<summary>Click to reveal answers</summary>

**Step 1:** z1 = [0.40, -0.40]
- z1[0] = 0.2 + 0.15 + 0.05 + 0.1 = 0.50
- z1[1] = -0.5 + 0.2 - 0.1 - 0.1 = -0.50

Wait, let me recalculate:
- z1[0] = 1.0*0.2 + 0.5*0.3 + (-0.5)*(-0.1) + 0.1 = 0.2 + 0.15 + 0.05 + 0.1 = 0.50
- z1[1] = 1.0*(-0.5) + 0.5*0.4 + (-0.5)*0.2 + (-0.1) = -0.5 + 0.2 - 0.1 - 0.1 = -0.50

**Step 2:** a1 = [0.50, 0.00]
- a1[0] = max(0, 0.50) = 0.50
- a1[1] = max(0, -0.50) = 0.00

**Step 3:** z2 = [0.40]
- z2 = 0.50*0.6 + 0.00*(-0.3) + 0.1 = 0.30 + 0.0 + 0.1 = 0.40

**Step 4:** y_hat = 0.5987
- y_hat = 1 / (1 + exp(-0.40)) = 1 / 1.6703 = 0.5987

**Classification:** 1 (since 0.5987 > 0.5)

</details>

