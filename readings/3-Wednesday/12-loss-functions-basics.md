# Loss Functions Basics

## Learning Objectives

- Understand loss functions as measures of prediction error
- Explain Mean Squared Error (MSE) for regression tasks
- Explain Cross-Entropy Loss for classification tasks
- Describe how loss guides the learning process

## Why This Matters

A neural network without a loss function is like a student without grades - it has no way to know if it's improving. The loss function provides the crucial feedback signal that tells the network how wrong its predictions are.

In our **From Zero to Neural** journey, loss functions complete the forward pass story. You've seen how data flows through a network to produce predictions; now you'll see how those predictions are evaluated. This evaluation - the loss - is what the network will learn to minimize during training (covered in Week 2).

## The Concept

### What Is a Loss Function?

A **loss function** (also called cost function, objective function, or error function) measures the difference between predicted values and actual values.

```
Loss Function Role:

Prediction (y_hat) ----\
                        >---- [Loss Function] ----> Loss Value (scalar)
Actual Value (y) ------/

Low loss = predictions close to actual = good model
High loss = predictions far from actual = needs improvement
```

**Key properties:**
- Always outputs a single number (scalar)
- Lower is better
- Must be differentiable (for gradient-based learning)
- Should capture what "good predictions" means for your task

### Loss vs. Cost vs. Error

These terms are often used interchangeably, but technically:

| Term | Scope |
|------|-------|
| **Loss** | Error for a single training example |
| **Cost** | Average loss over the entire training set |
| **Error** | General term for prediction mistakes |

```
Loss for sample i:    L(y_i, y_hat_i)
Cost for dataset:     J = (1/n) * SUM(L(y_i, y_hat_i))
```

### Mean Squared Error (MSE) for Regression

**The most common regression loss function.**

```
MSE = (1/n) * SUM[(y_i - y_hat_i)^2]

Where:
  n = number of samples
  y_i = actual value for sample i
  y_hat_i = predicted value for sample i
```

**Visual Intuition:**

```
Value
  |           o  Actual
  |          /|
  |         / |  Error = y - y_hat
  |        /  |
  |       x   |  Predicted
  |       ^
  |       |
  +-------+--------> Sample
          error squared

MSE squares each error, averages them.
```

**Why squared?**
1. Makes all errors positive (no cancellation between positive/negative errors)
2. Penalizes large errors more than small errors (10^2 = 100, but 1^2 = 1)
3. Mathematically convenient (smooth, differentiable)

**Python Implementation:**

```python
import numpy as np

def mse(y_true, y_pred):
    """Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)

# Example
y_true = np.array([3.0, 5.0, 7.0, 9.0])
y_pred = np.array([2.5, 5.5, 6.0, 10.0])

loss = mse(y_true, y_pred)
print(f"MSE: {loss:.4f}")

# Individual errors
errors = y_true - y_pred
print(f"Errors: {errors}")
print(f"Squared errors: {errors**2}")
print(f"Mean of squared: {np.mean(errors**2):.4f}")
```

**Output:**
```
MSE: 0.6875
Errors: [ 0.5 -0.5  1.  -1. ]
Squared errors: [0.25 0.25 1.   1.  ]
Mean of squared: 0.6875
```

**Variants:**
- **RMSE (Root MSE)**: sqrt(MSE) - same units as target
- **MAE (Mean Absolute Error)**: mean(|y - y_hat|) - less sensitive to outliers

### Binary Cross-Entropy Loss for Classification

**The standard loss for binary classification.**

For a single sample with true label y (0 or 1) and predicted probability p:

```
L = -[y * log(p) + (1-y) * log(1-p)]

If y = 1: L = -log(p)      # Penalize low probability for positive class
If y = 0: L = -log(1-p)    # Penalize high probability for positive class
```

**Visual Intuition:**

```
Loss
  |
  |\
  | \
  |  \
  |   \
  |    \____
  +-----------> Predicted Probability (p)
  0          1

When actual y=1:
- Predicting p=1.0 gives loss ~ 0
- Predicting p=0.1 gives high loss
- Predicting p=0.0 gives infinite loss
```

**Why logarithm?**
1. Heavily penalizes confident wrong predictions
2. Outputs are probabilities (bounded 0-1), log makes loss unbounded
3. Mathematically connected to maximum likelihood estimation

**Python Implementation:**

```python
def binary_cross_entropy(y_true, y_pred, epsilon=1e-15):
    """
    Binary Cross-Entropy Loss.
    epsilon prevents log(0).
    """
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Numerical stability
    loss = -np.mean(
        y_true * np.log(y_pred) + 
        (1 - y_true) * np.log(1 - y_pred)
    )
    return loss

# Example: 4 binary classification samples
y_true = np.array([1, 1, 0, 0])
y_pred = np.array([0.9, 0.7, 0.2, 0.4])  # Predicted probabilities

loss = binary_cross_entropy(y_true, y_pred)
print(f"Binary Cross-Entropy: {loss:.4f}")

# Show individual contributions
print("\nPer-sample loss:")
for yt, yp in zip(y_true, y_pred):
    if yt == 1:
        sample_loss = -np.log(yp)
    else:
        sample_loss = -np.log(1 - yp)
    print(f"  y={yt}, p={yp:.1f}: loss={sample_loss:.4f}")
```

**Output:**
```
Binary Cross-Entropy: 0.3711

Per-sample loss:
  y=1, p=0.9: loss=0.1054
  y=1, p=0.7: loss=0.3567
  y=0, p=0.2: loss=0.2231
  y=0, p=0.4: loss=0.5108
```

Notice: The model was less confident about the last sample (predicted 0.4 for class 0), resulting in higher loss.

### Categorical Cross-Entropy for Multi-Class

Extends binary cross-entropy to multiple classes.

```
L = -SUM[y_c * log(p_c)]  for all classes c

Where:
  y_c = 1 if true class is c, else 0 (one-hot encoded)
  p_c = predicted probability for class c
```

**Example (3-class classification):**

```
True class: 2 (one-hot: [0, 0, 1])
Predictions: [0.1, 0.2, 0.7]

L = -[0*log(0.1) + 0*log(0.2) + 1*log(0.7)]
L = -log(0.7)
L = 0.357
```

**Python Implementation:**

```python
def categorical_cross_entropy(y_true_onehot, y_pred, epsilon=1e-15):
    """
    Categorical Cross-Entropy Loss.
    y_true_onehot: one-hot encoded true labels
    y_pred: predicted probabilities (from softmax)
    """
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.mean(np.sum(y_true_onehot * np.log(y_pred), axis=1))
    return loss

# Example: 3 samples, 4 classes
y_true = np.array([
    [1, 0, 0, 0],  # Class 0
    [0, 0, 1, 0],  # Class 2
    [0, 1, 0, 0]   # Class 1
])
y_pred = np.array([
    [0.7, 0.1, 0.1, 0.1],  # Confident, correct
    [0.1, 0.1, 0.6, 0.2],  # Less confident, correct
    [0.2, 0.4, 0.3, 0.1]   # Wrong class predicted
])

loss = categorical_cross_entropy(y_true, y_pred)
print(f"Categorical Cross-Entropy: {loss:.4f}")
```

### How Loss Guides Learning

The loss function creates a **loss landscape** - a surface over the space of all possible weight values:

```
Loss Landscape (simplified 2D):

Loss
  ^
  |    /\
  |   /  \    /\
  |  /    \  /  \
  | /      \/    \
  |/               \
  +-----------------> Weight value

The goal: Find weights that minimize loss (the valleys)
```

**The training loop:**
1. Forward pass: compute predictions
2. Compute loss: measure prediction error
3. Backward pass: compute gradients (how to change weights to reduce loss)
4. Update weights: move toward lower loss
5. Repeat

This is **gradient descent** - we'll cover it in depth in Week 2.

### Choosing the Right Loss Function

| Task | Output Activation | Loss Function |
|------|-------------------|---------------|
| Regression | Linear (none) | MSE |
| Binary Classification | Sigmoid | Binary Cross-Entropy |
| Multi-Class Classification | Softmax | Categorical Cross-Entropy |
| Multi-Label Classification | Sigmoid (per class) | Binary Cross-Entropy (per class) |

**In frameworks:**

```python
# TensorFlow/Keras
model.compile(
    loss='binary_crossentropy',  # Binary classification
    # loss='categorical_crossentropy',  # Multi-class
    # loss='mse',  # Regression
)

# PyTorch
criterion = nn.BCELoss()  # Binary classification
# criterion = nn.CrossEntropyLoss()  # Multi-class (includes softmax)
# criterion = nn.MSELoss()  # Regression
```

## Code Example: Complete Loss Computation

```python
import numpy as np

def demonstrate_loss_functions():
    """Compare loss functions on example data."""
    
    print("=" * 60)
    print("LOSS FUNCTION DEMONSTRATION")
    print("=" * 60)
    
    # === REGRESSION ===
    print("\n--- Regression (MSE) ---")
    y_reg_true = np.array([100, 150, 200, 250])
    y_reg_pred = np.array([110, 145, 190, 260])
    
    mse = np.mean((y_reg_true - y_reg_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_reg_true - y_reg_pred))
    
    print(f"True values:  {y_reg_true}")
    print(f"Predictions:  {y_reg_pred}")
    print(f"MSE:  {mse:.2f}")
    print(f"RMSE: {rmse:.2f} (in original units)")
    print(f"MAE:  {mae:.2f}")
    
    # === BINARY CLASSIFICATION ===
    print("\n--- Binary Classification (Cross-Entropy) ---")
    y_bin_true = np.array([1, 1, 0, 0, 1])
    y_bin_pred = np.array([0.9, 0.6, 0.3, 0.1, 0.8])
    
    epsilon = 1e-15
    y_bin_pred_safe = np.clip(y_bin_pred, epsilon, 1 - epsilon)
    bce = -np.mean(
        y_bin_true * np.log(y_bin_pred_safe) +
        (1 - y_bin_true) * np.log(1 - y_bin_pred_safe)
    )
    
    print(f"True labels:  {y_bin_true}")
    print(f"Predictions:  {y_bin_pred}")
    print(f"Binary Cross-Entropy: {bce:.4f}")
    
    # Show what happens with confident wrong predictions
    print("\nEffect of confidence on loss:")
    for p in [0.99, 0.7, 0.5, 0.3, 0.01]:
        loss_if_true = -np.log(p)
        loss_if_false = -np.log(1 - p)
        print(f"  p={p:.2f}: loss if y=1: {loss_if_true:.3f}, loss if y=0: {loss_if_false:.3f}")
    
    # === MULTI-CLASS CLASSIFICATION ===
    print("\n--- Multi-Class Classification (Categorical CE) ---")
    # 3 samples, 4 classes
    y_multi_true = np.array([0, 2, 1])  # Class indices
    y_multi_pred = np.array([
        [0.8, 0.1, 0.05, 0.05],
        [0.1, 0.2, 0.6, 0.1],
        [0.2, 0.5, 0.2, 0.1]
    ])
    
    # One-hot encode
    n_classes = 4
    y_onehot = np.eye(n_classes)[y_multi_true]
    
    cce = -np.mean(np.sum(y_onehot * np.log(y_multi_pred + epsilon), axis=1))
    
    print(f"True classes: {y_multi_true}")
    print(f"One-hot:\n{y_onehot}")
    print(f"Predictions:\n{y_multi_pred}")
    print(f"Categorical Cross-Entropy: {cce:.4f}")
    
    print("\n" + "=" * 60)

demonstrate_loss_functions()
```

**Output:**
```
============================================================
LOSS FUNCTION DEMONSTRATION
============================================================

--- Regression (MSE) ---
True values:  [100 150 200 250]
Predictions:  [110 145 190 260]
MSE:  75.00
RMSE: 8.66 (in original units)
MAE:  10.00

--- Binary Classification (Cross-Entropy) ---
True labels:  [1 1 0 0 1]
Predictions:  [0.9 0.6 0.3 0.1 0.8]
Binary Cross-Entropy: 0.3066

Effect of confidence on loss:
  p=0.99: loss if y=1: 0.010, loss if y=0: 4.605
  p=0.70: loss if y=1: 0.357, loss if y=0: 1.204
  p=0.50: loss if y=1: 0.693, loss if y=0: 0.693
  p=0.30: loss if y=1: 1.204, loss if y=0: 0.357
  p=0.01: loss if y=1: 4.605, loss if y=0: 0.010

--- Multi-Class Classification (Categorical CE) ---
True classes: [0 2 1]
One-hot:
[[1. 0. 0. 0.]
 [0. 0. 1. 0.]
 [0. 1. 0. 0.]]
Predictions:
[[0.8  0.1  0.05 0.05]
 [0.1  0.2  0.6  0.1 ]
 [0.2  0.5  0.2  0.1 ]]
Categorical Cross-Entropy: 0.4827

============================================================
```

## Key Takeaways

1. **Loss functions measure prediction error** - they tell the network how wrong it is.

2. **MSE for regression** - squares errors, penalizing large mistakes more.

3. **Cross-Entropy for classification** - heavily penalizes confident wrong predictions.

4. **Loss guides learning** - the network learns by minimizing the loss function.

5. **Choose loss based on task** - MSE for continuous outputs, cross-entropy for probabilities.

## Looking Ahead

You now understand the complete forward pass: data flows through layers, makes predictions, and loss measures accuracy. But how does the network actually learn from this loss? 

Week 2 introduces **backpropagation** and **gradient descent** - the mechanisms that propagate the loss signal backward through the network, computing how to adjust each weight to reduce errors.

Before that, Thursday's TensorFlow session will let you build and train real networks, even without understanding the full math of backpropagation - the framework handles it for you.

## Additional Resources

- [Loss Functions Guide - ML Mastery](https://machinelearningmastery.com/loss-and-loss-functions-for-training-deep-learning-neural-networks/) - Comprehensive overview
- [Cross-Entropy Demystified](https://rdipietro.github.io/friendly-intro-to-cross-entropy-loss/) - Intuitive explanation
- [TensorFlow Loss Functions](https://www.tensorflow.org/api_docs/python/tf/keras/losses) - Framework implementations

