# Exercise 04: Build a Perceptron from Scratch

## Learning Objectives

- Implement a perceptron class with all core components
- Understand weight initialization, forward pass, and learning rule
- Visualize decision boundaries
- Discover the limits of single-layer networks

## Duration

**Estimated Time:** 90 minutes

## Background

In `demo_04_perceptron_logic_gates.py`, we saw a perceptron learn AND/OR gates. Now you'll build your own `Perceptron` class from the ground up.

---

## The Perceptron Math

```
Forward Pass:
    z = w1*x1 + w2*x2 + ... + wn*xn + b
    z = W . X + b
    y = activation(z)

Learning Rule (for step activation):
    error = y_true - y_pred
    w_new = w_old + learning_rate * error * x
    b_new = b_old + learning_rate * error
```

---

## Part 1: Build the Perceptron Class (45 min)

### Task 1.1: Class Skeleton

Navigate to `starter_code/perceptron.py`:

```python
import numpy as np

class Perceptron:
    """A single perceptron (binary classifier)."""
    
    def __init__(self, n_features, learning_rate=0.1, activation='step'):
        """
        Initialize the perceptron.
        
        Args:
            n_features: Number of input features
            learning_rate: Step size for weight updates
            activation: 'step' or 'sigmoid'
        """
        # TODO: Initialize weights with small random values
        self.weights = None  # Shape: (n_features,)
        
        # TODO: Initialize bias to 0
        self.bias = None
        
        self.learning_rate = learning_rate
        self.activation_name = activation
        
    def _activation(self, z):
        """
        Apply activation function.
        
        Args:
            z: Weighted sum (can be scalar or array)
        
        Returns:
            Activated output
        """
        if self.activation_name == 'step':
            # TODO: Return 1 if z >= 0, else 0
            pass
        elif self.activation_name == 'sigmoid':
            # TODO: Return 1 / (1 + exp(-z))
            pass
    
    def forward(self, X):
        """
        Compute forward pass.
        
        Args:
            X: Input features (n_samples, n_features) or (n_features,)
        
        Returns:
            Predictions
        """
        # TODO: Compute z = X @ weights + bias
        # TODO: Apply activation
        pass
    
    def predict(self, X):
        """Make predictions (alias for forward)."""
        output = self.forward(X)
        if self.activation_name == 'sigmoid':
            return (output >= 0.5).astype(int)
        return output.astype(int)
    
    def fit(self, X, y, epochs=100):
        """
        Train the perceptron.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
            epochs: Number of passes through data
        
        Returns:
            self, history
        """
        history = {'errors': [], 'weights': [], 'bias': []}
        
        for epoch in range(epochs):
            errors = 0
            
            for xi, yi in zip(X, y):
                # TODO: Get prediction
                prediction = None
                
                # TODO: Calculate error
                error = None
                
                # TODO: Update weights and bias if error != 0
                if error != 0:
                    # w = w + lr * error * x
                    # b = b + lr * error
                    pass
                    errors += 1
            
            # Record history
            history['errors'].append(errors)
            history['weights'].append(self.weights.copy())
            history['bias'].append(self.bias)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Errors = {errors}")
            
            # Early stopping if no errors
            if errors == 0:
                print(f"Converged at epoch {epoch}!")
                break
        
        return self, history
```

### Task 1.2: Test with AND Gate

```python
# AND gate data
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

# TODO: Create and train perceptron
perceptron = Perceptron(n_features=2, learning_rate=0.1)
perceptron, history = perceptron.fit(X_and, y_and, epochs=100)

# TODO: Test predictions
predictions = perceptron.predict(X_and)
print(f"Predictions: {predictions}")
print(f"Actual: {y_and}")
print(f"Accuracy: {(predictions == y_and).mean():.2%}")
```

### Task 1.3: Test with OR Gate

```python
# OR gate data
X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_or = np.array([0, 1, 1, 1])

# TODO: Create and train new perceptron for OR
```

---

## Part 2: Visualization (25 min)

### Task 2.1: Plot Decision Boundary

```python
def plot_decision_boundary(perceptron, X, y, title):
    """
    Visualize the decision boundary.
    
    The decision boundary is where: w1*x1 + w2*x2 + b = 0
    Solving for x2: x2 = -(w1*x1 + b) / w2
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(8, 6))
    
    # TODO: Plot data points
    # Red circles for class 0, green squares for class 1
    
    # TODO: Plot decision boundary line
    # x1_range = np.linspace(-0.5, 1.5, 100)
    # x2_boundary = -(w1 * x1_range + b) / w2
    
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# TODO: Call for AND and OR gates
```

### Task 2.2: Learning Curve

```python
# TODO: Plot errors over epochs
# plt.plot(history['errors'])
# plt.title('Learning Curve')
# plt.xlabel('Epoch')
# plt.ylabel('Errors')
```

---

## Part 3: The XOR Challenge (20 min)

### Task 3.1: Try XOR

```python
# XOR gate data
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

# TODO: Train perceptron on XOR
perceptron_xor = Perceptron(n_features=2, learning_rate=0.1)
perceptron_xor, history_xor = perceptron_xor.fit(X_xor, y_xor, epochs=1000)

# TODO: Check accuracy
```

### Task 3.2: Explain the Failure

Answer in comments:

```python
# Q1: Did the perceptron converge for XOR? What was the final accuracy?
# Answer:

# Q2: Why can't a single perceptron learn XOR? (Hint: linear separability)
# Answer:

# Q3: What would you need to solve XOR? (Preview of multi-layer networks)
# Answer:
```

---

## Bonus Challenges

### Challenge A: Sigmoid Activation

Implement and test with sigmoid activation:

```python
perceptron_sigmoid = Perceptron(n_features=2, activation='sigmoid')
# Test on AND gate - does it still work?
```

### Challenge B: Multi-Class (One-vs-Rest)

Create 3 perceptrons to classify the Iris dataset (3 classes):

```python
# Hint: Train one perceptron per class
# Perceptron 1: Class 0 vs not-Class-0
# Perceptron 2: Class 1 vs not-Class-1
# Perceptron 3: Class 2 vs not-Class-2
```

---

## Definition of Done

- [ ] Perceptron class fully implemented
- [ ] Successfully learns AND gate (100% accuracy)
- [ ] Successfully learns OR gate (100% accuracy)
- [ ] Decision boundary visualized for both
- [ ] XOR attempted and failure explained
- [ ] All reflection questions answered

---

## Key Takeaways

After completing this exercise, you should understand:

1. **Weights** determine the importance of each input
2. **Bias** shifts the decision boundary
3. **Learning rule** updates weights based on errors
4. **Linear separability** is the limit of single-layer networks
5. **Multi-layer networks** are needed for non-linear problems (XOR)

