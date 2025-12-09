# Multi-Layer Perceptrons (MLPs)

## Learning Objectives

- Explain the architecture of Multi-Layer Perceptrons: input, hidden, and output layers
- Understand how MLPs overcome single perceptron limitations (XOR problem)
- Develop intuition for the Universal Approximation Theorem
- Apply guidelines for choosing layer sizes and network depth

## Why This Matters

The Multi-Layer Perceptron is where neural networks become genuinely powerful. By stacking layers of neurons, we transform simple linear classifiers into systems that can approximate any continuous function - recognizing patterns, making decisions, and solving problems that seemed impossible to the AI researchers of the 1960s.

In our **From Zero to Neural** journey, the MLP is the first "real" neural network architecture you'll learn. It's the foundation for everything from image classifiers to language models, and understanding it unlocks your ability to build practical AI systems starting Thursday with TensorFlow.

## The Concept

### From Single Perceptron to Multi-Layer Network

Recall that a single perceptron can only learn linearly separable patterns:

```
Single Perceptron:

   Input -> [Perceptron] -> Output

Can solve:               Cannot solve:
    AND, OR                  XOR

  o o | x x              o      x
  o o | x x                 x      o
      |                  (no line separates)
  (line separates)
```

The solution: **stack perceptrons into layers**.

### MLP Architecture

```
MULTI-LAYER PERCEPTRON

Input Layer        Hidden Layer(s)        Output Layer
-----------        ---------------        ------------

   x1 ----\        /---[h1]---\           /----> y1
           \------/            \---------/
   x2 ----\ \----/---[h2]---\ / \-------/-----> y2
           \ \  /            X   \-----/
   x3 ----\ \ \/   /---[h3]--/\   \---/-------> y3
           \ \/\  /            \   \_/
   x4 ------\/ \ /---[h4]-------\-/
              \ /
               X
          (Fully Connected)

Input:  Raw features (pixels, measurements, etc.)
Hidden: Learned representations (increasingly abstract)
Output: Final predictions (classes, values, etc.)
```

**Key terminology:**

| Term | Definition |
|------|------------|
| **Layer** | Collection of neurons at the same depth |
| **Input layer** | Receives raw features (not actually neurons) |
| **Hidden layer** | Intermediate layer(s) between input and output |
| **Output layer** | Produces final predictions |
| **Dense/Fully connected** | Every neuron connects to every neuron in next layer |
| **Width** | Number of neurons in a layer |
| **Depth** | Number of hidden layers |

### Solving XOR with Two Layers

Let's see how an MLP with one hidden layer solves XOR:

```
XOR Truth Table:
x1  x2  |  y
0   0   |  0
0   1   |  1
1   0   |  1
1   1   |  0
```

**The trick**: The hidden layer creates new features that ARE linearly separable.

```
Step 1: Hidden layer transforms the space

Original Space:           Hidden Layer Space:
    x2                        h2
     |                         |
   1 |  0      1               |    (0,1) --> 1
     |                       1 |    *      *
   0 |  1      0               |
     +---------> x1          0 |    *      *
        0      1               +-----------> h1

The hidden layer learns:
h1 = OR(x1, x2)   --> 1 if either input is 1
h2 = NAND(x1, x2) --> 0 only if both inputs are 1

h1  h2  |  y (XOR)
0   1   |  0    (both inputs were 0)
1   1   |  1    (one input was 1)  
1   1   |  1    (one input was 1)
1   0   |  0    (both inputs were 1)

Step 2: Output layer separates in new space

In (h1, h2) space, the classes ARE linearly separable!
y = AND(h1, h2)
```

**The insight**: Hidden layers transform data into representations where the problem becomes linearly separable.

### The Universal Approximation Theorem

One of the most profound results in neural network theory:

> **Theorem**: A feedforward network with a single hidden layer containing a finite number of neurons can approximate any continuous function on compact subsets of R^n, under mild assumptions on the activation function.

**In plain English**: Given enough neurons in one hidden layer, an MLP can learn any continuous input-output mapping.

```
Any continuous function:     Can be approximated by MLP:

f(x)                         Network with enough hidden neurons
   |    _____                    _____
   |   /     \                  /     \
   |  /       \                /       \
   | /         ----           /         ----
   |/                        /
   +-------------> x        +-------------> x
                            
                            (with sufficient width)
```

**Caveats:**
- Theorem says it's possible, not that training will find the solution
- May need impractically many neurons in one layer
- Deeper networks often more efficient than one very wide layer

### Width vs. Depth

**Wide networks (few layers, many neurons):**
```
Input -> [Many neurons] -> Output
```
- Can theoretically approximate any function
- May need impractically many neurons
- Harder to train effectively

**Deep networks (many layers, moderate width):**
```
Input -> [Layer] -> [Layer] -> [Layer] -> ... -> Output
```
- Learn hierarchical representations
- More parameter-efficient
- Can suffer from vanishing gradients (solved by ReLU, residual connections)

**Empirical finding**: Deep networks often outperform wide networks with similar parameter counts.

### Layer Sizing Guidelines

**Input layer:**
- Size = number of input features
- No neurons - just passes data forward

**Hidden layers:**
```
Rules of thumb (starting points):
- Start with 1-2 hidden layers for simple problems
- Width: between input size and output size, or 2-3x input size
- Powers of 2 are convenient: 32, 64, 128, 256, 512
- More complex patterns may need more layers/neurons
```

**Output layer:**
- Size = number of outputs (classes for classification, 1 for single regression)
- Activation depends on task:
  - Binary classification: 1 neuron + sigmoid
  - Multi-class classification: N neurons + softmax
  - Regression: 1 neuron + linear (no activation)

### Common Architectures

**Binary Classification:**
```
Input(features) -> Dense(64, ReLU) -> Dense(32, ReLU) -> Dense(1, Sigmoid)
```

**Multi-Class Classification (10 classes):**
```
Input(features) -> Dense(128, ReLU) -> Dense(64, ReLU) -> Dense(10, Softmax)
```

**Regression:**
```
Input(features) -> Dense(64, ReLU) -> Dense(32, ReLU) -> Dense(1, Linear)
```

## Code Example: MLP from Scratch

```python
import numpy as np

class MLP:
    """A simple Multi-Layer Perceptron."""
    
    def __init__(self, layer_sizes):
        """
        Initialize network.
        layer_sizes: list of neurons per layer, e.g., [4, 8, 4, 2]
                     (input_dim, hidden1, hidden2, output_dim)
        """
        self.layers = []
        self.activations = []
        
        # Initialize weights and biases for each layer
        for i in range(len(layer_sizes) - 1):
            # Xavier initialization
            W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.layers.append({'W': W, 'b': b})
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def forward(self, X):
        """Forward pass through all layers."""
        self.activations = [X]  # Store for backprop
        
        current = X
        for i, layer in enumerate(self.layers):
            z = current @ layer['W'] + layer['b']
            
            # ReLU for hidden layers, sigmoid for output
            if i < len(self.layers) - 1:
                current = self.relu(z)
            else:
                current = self.sigmoid(z)
            
            self.activations.append(current)
        
        return current
    
    def predict(self, X):
        """Make predictions (0 or 1 for binary classification)."""
        probs = self.forward(X)
        return (probs >= 0.5).astype(int)

# Create XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Create MLP: 2 inputs -> 4 hidden -> 1 output
mlp = MLP([2, 4, 1])

# Before training (random weights)
print("Before training:")
predictions = mlp.forward(X)
for xi, yi, pred in zip(X, y, predictions):
    print(f"  Input: {xi}, Target: {yi[0]}, Prediction: {pred[0]:.3f}")

# Note: This example doesn't include training (backpropagation)
# We'll cover that in Week 2. For now, see how the architecture works.
```

**Sample Output:**
```
Before training:
  Input: [0 0], Target: 0, Prediction: 0.612
  Input: [0 1], Target: 1, Prediction: 0.687
  Input: [1 0], Target: 1, Prediction: 0.623
  Input: [1 1], Target: 0, Prediction: 0.701

(Random weights = random predictions)
```

### Using scikit-learn's MLP

```python
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Create and train MLP
mlp = MLPClassifier(
    hidden_layer_sizes=(4,),  # One hidden layer with 4 neurons
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42
)

mlp.fit(X, y)

# Evaluate
predictions = mlp.predict(X)
print("XOR Results with sklearn MLP:")
for xi, yi, pred in zip(X, y, predictions):
    print(f"  Input: {xi}, Target: {yi}, Prediction: {pred}")
print(f"Accuracy: {accuracy_score(y, predictions):.1%}")
```

**Output:**
```
XOR Results with sklearn MLP:
  Input: [0 0], Target: 0, Prediction: 0
  Input: [0 1], Target: 1, Prediction: 1
  Input: [1 0], Target: 1, Prediction: 1
  Input: [1 1], Target: 0, Prediction: 0
Accuracy: 100.0%
```

The MLP successfully learned XOR.

### Information Flow in MLPs

```
Forward Pass (Inference):
   Input -> [Layer 1] -> [Layer 2] -> ... -> [Output Layer] -> Prediction
          ReLU      ReLU              Sigmoid

Each layer:
   z = W @ previous_activation + b
   a = activation(z)

The network transforms:
   Raw features -> Learned features -> More abstract features -> Decision
```

## Key Takeaways

1. **MLPs stack layers of neurons** to learn complex, non-linear patterns that single perceptrons cannot.

2. **Hidden layers create new representations** - transforming data into spaces where problems become linearly separable.

3. **Universal Approximation**: With enough neurons, an MLP can approximate any continuous function.

4. **Depth vs. Width**: Deep networks (more layers) are often more efficient than wide networks (more neurons per layer).

5. **Architecture choices matter**: Number of layers, neurons per layer, and activation functions all affect learning.

## Looking Ahead

You understand the architecture - but how does data actually flow through these layers mathematically? The next reading on **Forward Propagation** walks through the exact computations step-by-step.

Then **Loss Functions** explains how networks measure their prediction errors - the signal that drives learning.

By Thursday, you'll implement these concepts in TensorFlow, building real MLPs that classify images.

## Additional Resources

- [Neural Network Playground](https://playground.tensorflow.org/) - Interactive MLP visualization
- [Universal Approximation Theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem) - Mathematical background
- [How to Configure Neural Network Layers](https://machinelearningmastery.com/how-to-configure-the-number-of-layers-and-nodes-in-a-neural-network/) - Practical guidelines

