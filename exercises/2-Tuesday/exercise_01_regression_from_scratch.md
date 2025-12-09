# Exercise 01: Linear Regression from Scratch

## Learning Objectives

- Implement gradient descent algorithm manually
- Understand the mathematical foundation of linear regression
- Visualize the learning process as weights update
- Compare your implementation to scikit-learn

## Duration

**Estimated Time:** 90 minutes

## Background

In `demo_01_regression_playground.py`, we used scikit-learn's `LinearRegression` class. But what happens inside that black box? In this exercise, you'll implement linear regression using gradient descent - the same optimization technique used in neural networks.

## The Math You Need

### Linear Regression Equation
```
y_pred = w * x + b

Where:
  w = weight (slope)
  b = bias (intercept)
  x = input feature
  y_pred = prediction
```

### Mean Squared Error (MSE)
```
MSE = (1/n) * SUM((y_pred - y_actual)^2)
```

### Gradient Descent Updates
```
w_new = w_old - learning_rate * dMSE/dw
b_new = b_old - learning_rate * dMSE/db

Where:
  dMSE/dw = (2/n) * SUM((y_pred - y_actual) * x)
  dMSE/db = (2/n) * SUM(y_pred - y_actual)
```

---

## Part 1: Implement the Building Blocks (30 min)

### Task 1.1: Create the Data

Navigate to `starter_code/regression_scratch.py` and complete the data generation:

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate training data
np.random.seed(42)
X = np.random.uniform(0, 10, 100)  # 100 points between 0 and 10
y = 2.5 * X + 7 + np.random.normal(0, 2, 100)  # True relationship: y = 2.5x + 7 + noise

# TODO: Visualize the data
# Your code here
```

### Task 1.2: Implement Prediction Function

```python
def predict(X, w, b):
    """
    Compute predictions using linear equation.
    
    Args:
        X: Input features (array)
        w: Weight (scalar)
        b: Bias (scalar)
    
    Returns:
        Predictions (array)
    """
    # TODO: Implement y = w * X + b
    pass
```

### Task 1.3: Implement MSE Loss

```python
def compute_mse(y_true, y_pred):
    """
    Compute Mean Squared Error.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
    
    Returns:
        MSE (scalar)
    """
    # TODO: Implement MSE = (1/n) * sum((y_pred - y_true)^2)
    pass
```

### Task 1.4: Implement Gradient Computation

```python
def compute_gradients(X, y_true, y_pred):
    """
    Compute gradients for w and b.
    
    Args:
        X: Input features
        y_true: Actual values
        y_pred: Predicted values
    
    Returns:
        (dw, db): Gradients for weight and bias
    """
    n = len(X)
    # TODO: Compute dw = (2/n) * sum((y_pred - y_true) * X)
    # TODO: Compute db = (2/n) * sum(y_pred - y_true)
    pass
```

---

## Part 2: Implement Training Loop (30 min)

### Task 2.1: Complete the Gradient Descent Function

```python
def gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    """
    Train linear regression using gradient descent.
    
    Args:
        X: Input features
        y: Target values
        learning_rate: Step size for updates
        epochs: Number of iterations
    
    Returns:
        w, b, history: Final weights and training history
    """
    # Initialize weights randomly
    w = np.random.randn() * 0.01
    b = 0.0
    
    history = {'loss': [], 'w': [], 'b': []}
    
    for epoch in range(epochs):
        # TODO: Step 1 - Make predictions
        y_pred = None  # Your code
        
        # TODO: Step 2 - Compute loss
        loss = None  # Your code
        
        # TODO: Step 3 - Compute gradients
        dw, db = None, None  # Your code
        
        # TODO: Step 4 - Update weights
        w = None  # Your code
        b = None  # Your code
        
        # Record history
        history['loss'].append(loss)
        history['w'].append(w)
        history['b'].append(b)
        
        # Print progress every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}, w = {w:.4f}, b = {b:.4f}")
    
    return w, b, history
```

### Task 2.2: Train Your Model

```python
# Train the model
w_final, b_final, history = gradient_descent(X, y, learning_rate=0.01, epochs=1000)

print(f"\nFinal Model: y = {w_final:.4f} * x + {b_final:.4f}")
print(f"True Relationship: y = 2.5 * x + 7")
```

---

## Part 3: Visualization (15 min)

### Task 3.1: Plot the Loss Curve

```python
# TODO: Plot how loss decreases over epochs
plt.figure(figsize=(10, 4))
# Your code here
plt.title('Training Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.show()
```

### Task 3.2: Plot the Final Fit

```python
# TODO: Plot data points and the learned regression line
plt.figure(figsize=(10, 6))
# Your code here
plt.title('Linear Regression from Scratch')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
```

### Task 3.3: Animate Weight Evolution (Bonus)

```python
# TODO (BONUS): Create an animation showing how the line changes during training
# Hint: Plot every 50th epoch's line
```

---

## Part 4: Compare with scikit-learn (15 min)

### Task 4.1: Train scikit-learn Model

```python
from sklearn.linear_model import LinearRegression

# TODO: Train sklearn model and compare weights
sklearn_model = LinearRegression()
# Your code here

print(f"Your Model:    w = {w_final:.4f}, b = {b_final:.4f}")
print(f"sklearn Model: w = {sklearn_model.coef_[0]:.4f}, b = {sklearn_model.intercept_:.4f}")
```

### Task 4.2: Compare Predictions

```python
# TODO: Compare predictions from both models
# Are they similar? What explains any differences?
```

---

## Reflection Questions

Answer these in comments at the end of your code:

1. **What happens if you increase the learning rate to 0.1? To 1.0?** Try it and describe what you observe.

2. **What happens if you decrease the learning rate to 0.0001?** How does this affect training?

3. **Why do we initialize weights with small random values instead of zeros?**

4. **How close did your implementation get to the true values (w=2.5, b=7)?** What factors affect this?

---

## Definition of Done

- [ ] All four functions implemented and working
- [ ] Model trains without errors
- [ ] Loss curve shows decreasing loss
- [ ] Final weights are close to true values (w near 2.5, b near 7)
- [ ] Comparison with sklearn shows similar results
- [ ] All reflection questions answered

---

## Solution

The solution is available in `solutions/exercise_01_solution.py`. Only check after completing your attempt!

