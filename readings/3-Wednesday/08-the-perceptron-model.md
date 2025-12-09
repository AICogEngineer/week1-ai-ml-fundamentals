# The Perceptron Model

## Learning Objectives

- Explain the perceptron as the foundational unit of neural networks
- Describe each component: inputs, weights, weighted sum, bias, and activation
- Interpret the perceptron geometrically as defining a decision boundary (hyperplane)
- Understand the mathematical notation used in neural network literature

## Why This Matters

The perceptron is where it all began. Invented by Frank Rosenblatt in 1958, this simple algorithm sparked the first wave of AI enthusiasm - and later, the first AI winter when its limitations were exposed.

In our **From Zero to Neural** journey, the perceptron is your atomic building block. Every neuron in every layer of every deep neural network - from the models recognizing your face to the ones generating images - is fundamentally a perceptron with better activation functions and training algorithms.

Master the perceptron, and you've mastered the foundation.

## The Concept

### The Perceptron: A Complete Breakdown

The perceptron is a binary linear classifier that makes decisions by computing a weighted sum of inputs and comparing it to a threshold.

**Complete Architecture:**

```
                    PERCEPTRON
        
   Inputs        Weights       Sum           Activation      Output
   ------        -------       ---           ----------      ------
   
    x1 -------- w1 ----\
                        \
    x2 -------- w2 -------> [Sigma] + b --> [activation] --> y
                        /       
    x3 -------- w3 ----/     (weighted       (decision        (0 or 1)
                              sum + bias)     function)
    
   (features)   (learned)    (linear        (non-linear)    (prediction)
                              combination)
```

### Component Deep Dive

#### 1. Inputs (x)

The input vector contains the features for a single example:

```
x = [x1, x2, x3, ..., xn]

Example (loan approval):
x = [income, credit_score, debt_ratio, employment_years]
x = [50000, 720, 0.3, 5]
```

#### 2. Weights (w)

Each input has an associated weight that determines its importance:

```
w = [w1, w2, w3, ..., wn]

Interpretation:
- Large positive weight: input strongly pushes toward positive class
- Large negative weight: input strongly pushes toward negative class
- Weight near zero: input has little influence
```

**Learning IS adjusting weights** - the perceptron learns by finding the right weight values.

#### 3. Weighted Sum (z)

The weighted sum combines all inputs:

```
z = w1*x1 + w2*x2 + w3*x3 + ... + wn*xn

Vector notation:
z = w . x = w^T * x (dot product)
```

**Python implementation:**
```python
import numpy as np

x = np.array([50000, 720, 0.3, 5])
w = np.array([0.00001, 0.01, -2, 0.1])

z = np.dot(w, x)  # Dot product
print(f"Weighted sum: {z}")  # Output: 50000*0.00001 + 720*0.01 + 0.3*(-2) + 5*0.1
```

#### 4. Bias (b)

The bias is an additional parameter that shifts the decision boundary:

```
z = (w . x) + b

The bias allows the perceptron to:
- Shift the decision boundary away from the origin
- Make predictions even when all inputs are zero
- Add flexibility to the model
```

**Intuition**: If the weighted sum of features suggests "approve," but there's a general policy to be cautious, a negative bias can shift toward "deny."

#### 5. Activation Function

The activation function converts the continuous weighted sum into a discrete decision:

**Step Function (Original Perceptron):**

```
        Output
          |
        1 |      ___________
          |     |
          |     |
        0 |_____|
          |
          +------|---------> z
                 0 (threshold)

f(z) = 1  if z >= 0
f(z) = 0  if z < 0
```

```python
def step_activation(z):
    return 1 if z >= 0 else 0
```

### The Complete Perceptron Formula

Putting it all together:

```
y = activation(w1*x1 + w2*x2 + ... + wn*xn + b)
y = activation(w . x + b)
y = activation(z)

For step activation:
y = 1  if (w . x + b) >= 0
y = 0  otherwise
```

### Geometric Interpretation: The Decision Boundary

The perceptron defines a **hyperplane** that separates two classes:

**In 2D (two features):**

```
The equation w1*x1 + w2*x2 + b = 0 defines a LINE

    x2
     |
   5 |  o  o  o
     |  o  o /
   3 |  o  /  x  x
     |   /   x  x
   1 | /  x  x  x
     |/
     +--------------> x1
       1  3  5  7

The line w1*x1 + w2*x2 + b = 0 separates:
- Class 1 (o): points where w.x + b >= 0
- Class 0 (x): points where w.x + b < 0
```

**The weight vector is perpendicular to the decision boundary:**

```
        ^
        |  w (weight vector)
        |
        |
  ------+------  Decision boundary
        |
        |
```

**The bias shifts the boundary:**

```
b = 0:          b > 0:           b < 0:
    |               |                 |
    |               |                 |
----+----       ----+------>      <---+----
    |               |                 |
    |               |                 |
(through origin)  (shifted right)   (shifted left)
```

**In 3D (three features):** The boundary is a PLANE

**In nD (n features):** The boundary is a HYPERPLANE

### Mathematical Notation Standards

You'll see various notations in literature:

| Notation | Meaning |
|----------|---------|
| \( x_i \) | i-th input feature |
| \( w_i \) | Weight for i-th feature |
| \( b \) or \( w_0 \) | Bias term |
| \( z \) | Weighted sum (pre-activation) |
| \( a \) or \( \hat{y} \) | Activation output (prediction) |
| \( y \) | True label |
| \( \sigma \) | Activation function (often sigmoid) |

**Matrix notation for batch processing:**

```
X = [batch_size x num_features]    # Multiple examples
W = [num_features x 1]             # Weights as column vector
b = scalar                          # Bias

Z = X @ W + b                       # Matrix multiplication
Y = activation(Z)                   # Element-wise activation
```

## Code Example: Perceptron from Scratch

```python
import numpy as np

class Perceptron:
    """A single perceptron (binary classifier)."""
    
    def __init__(self, num_features, learning_rate=0.1):
        """Initialize with random weights and zero bias."""
        self.weights = np.random.randn(num_features) * 0.01
        self.bias = 0.0
        self.learning_rate = learning_rate
    
    def activation(self, z):
        """Step function activation."""
        return np.where(z >= 0, 1, 0)
    
    def predict(self, X):
        """Forward pass: compute predictions."""
        z = np.dot(X, self.weights) + self.bias
        return self.activation(z)
    
    def train(self, X, y, epochs=100):
        """Train using the perceptron learning rule."""
        for epoch in range(epochs):
            errors = 0
            for xi, yi in zip(X, y):
                # Make prediction
                prediction = self.predict(xi.reshape(1, -1))[0]
                
                # Calculate error
                error = yi - prediction
                
                if error != 0:
                    errors += 1
                    # Update weights: w = w + lr * error * x
                    self.weights += self.learning_rate * error * xi
                    self.bias += self.learning_rate * error
            
            if errors == 0:
                print(f"Converged at epoch {epoch}")
                break
        
        return self

# Example: AND gate
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([0, 0, 0, 1])  # AND truth table

# Train perceptron
perceptron = Perceptron(num_features=2)
perceptron.train(X, y, epochs=100)

# Test
print("\nAND Gate Results:")
print(f"Weights: {perceptron.weights}")
print(f"Bias: {perceptron.bias}")
for xi, yi in zip(X, y):
    pred = perceptron.predict(xi.reshape(1, -1))[0]
    print(f"Input: {xi} -> Predicted: {pred}, Actual: {yi}")
```

**Sample Output:**
```
Converged at epoch 6

AND Gate Results:
Weights: [0.2 0.1]
Bias: -0.3
Input: [0 0] -> Predicted: 0, Actual: 0
Input: [0 1] -> Predicted: 0, Actual: 0
Input: [1 0] -> Predicted: 0, Actual: 0
Input: [1 1] -> Predicted: 1, Actual: 1
```

### Visualizing the Decision Boundary

```python
import matplotlib.pyplot as plt

def plot_decision_boundary(perceptron, X, y):
    """Visualize the perceptron's decision boundary."""
    # Create mesh grid
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    
    # Decision boundary line: w1*x1 + w2*x2 + b = 0
    # Solving for x2: x2 = -(w1*x1 + b) / w2
    x1_line = np.linspace(x_min, x_max, 100)
    x2_line = -(perceptron.weights[0] * x1_line + perceptron.bias) / perceptron.weights[1]
    
    plt.figure(figsize=(8, 6))
    plt.plot(x1_line, x2_line, 'g-', label='Decision Boundary', linewidth=2)
    
    # Plot points
    for i, (xi, yi) in enumerate(zip(X, y)):
        color = 'blue' if yi == 1 else 'red'
        marker = 'o' if yi == 1 else 'x'
        plt.scatter(xi[0], xi[1], c=color, marker=marker, s=200)
    
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Perceptron Decision Boundary (AND Gate)')
    plt.legend()
    plt.grid(True)
    plt.show()
```

### The XOR Problem: Why One Perceptron Isn't Enough

The perceptron has a critical limitation discovered by Minsky and Papert (1969):

```
XOR Truth Table:
x1  x2  |  y
-----------
0   0   |  0
0   1   |  1
1   0   |  1
1   1   |  0

Visualization:
    x2
     |
   1 |  x      o      No single line can separate
     |                the o's from the x's!
   0 |  o      x
     |
     +-------------> x1
        0      1
```

A single perceptron can only create linear decision boundaries. XOR requires a non-linear boundary.

**Solution**: Stack multiple perceptrons into layers (Multi-Layer Perceptron), which you'll learn about later today.

## The Perceptron Learning Rule

The perceptron updates weights based on errors:

```
For each misclassified point:
    w_new = w_old + learning_rate * error * x
    b_new = b_old + learning_rate * error

Where:
    error = actual - predicted  (either -1, 0, or +1)
```

**Convergence Theorem**: If the data is linearly separable, the perceptron learning algorithm will converge to a solution in finite steps.

## Key Takeaways

1. **The perceptron is the fundamental unit** - a weighted sum plus bias, passed through an activation function.

2. **Weights encode learned importance** - larger weights mean more influence on the output.

3. **The decision boundary is a hyperplane** - the perceptron divides space into two regions.

4. **Single perceptrons are limited** - they can only learn linearly separable patterns (not XOR).

5. **The perceptron learning rule is simple** - adjust weights in the direction of the error.

## Looking Ahead

You've seen that single perceptrons use a step function for activation. The next reading on **Activation Functions** explores why this is limiting and introduces smooth alternatives like sigmoid and ReLU that enable gradient-based learning.

The reading after that on **Multi-Layer Perceptrons** shows how stacking perceptrons overcomes the XOR problem and enables learning any function.

## Additional Resources

- [The Perceptron - Original Paper](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.335.3398&rep=rep1&type=pdf) - Rosenblatt's 1958 paper
- [Perceptron Visualization](https://ml-playground.com/#perceptron) - Interactive demo
- [Why the XOR Problem Matters](https://medium.com/@lucasheld/why-the-xor-problem-is-important-to-understand-ai-f75a59f6c35d) - Historical context

