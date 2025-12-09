"""
Exercise 04: Build a Perceptron from Scratch
============================================

Complete the Perceptron class implementation.
"""

import numpy as np
import matplotlib.pyplot as plt


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
        np.random.seed(42)
        
        # TODO: Initialize weights with small random values
        self.weights = None  # Shape: (n_features,)
        
        # TODO: Initialize bias to 0
        self.bias = None
        
        self.learning_rate = learning_rate
        self.activation_name = activation
        
    def _activation(self, z):
        """Apply activation function."""
        if self.activation_name == 'step':
            # TODO: Return 1 if z >= 0, else 0
            pass
        elif self.activation_name == 'sigmoid':
            # TODO: Return 1 / (1 + exp(-z))
            pass
    
    def forward(self, X):
        """Compute forward pass."""
        # Handle both single sample and batch
        X = np.atleast_2d(X)
        
        # TODO: Compute z = X @ weights + bias
        z = None
        
        # TODO: Apply activation
        return self._activation(z)
    
    def predict(self, X):
        """Make predictions (alias for forward)."""
        output = self.forward(X)
        if self.activation_name == 'sigmoid':
            return (output >= 0.5).astype(int).flatten()
        return output.astype(int).flatten()
    
    def fit(self, X, y, epochs=100):
        """Train the perceptron."""
        history = {'errors': [], 'weights': [], 'bias': []}
        
        for epoch in range(epochs):
            errors = 0
            
            for xi, yi in zip(X, y):
                # TODO: Get prediction (single sample)
                prediction = None
                
                # TODO: Calculate error
                error = None
                
                # TODO: Update weights and bias if error != 0
                if error != 0:
                    # w = w + lr * error * x
                    # b = b + lr * error
                    errors += 1
            
            history['errors'].append(errors)
            history['weights'].append(self.weights.copy())
            history['bias'].append(self.bias)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Errors = {errors}")
            
            if errors == 0:
                print(f"Converged at epoch {epoch}!")
                break
        
        return self, history


def plot_decision_boundary(perceptron, X, y, title):
    """Visualize the decision boundary."""
    plt.figure(figsize=(8, 6))
    
    # TODO: Plot data points
    # for label, marker, color in [(0, 'o', 'red'), (1, 's', 'green')]:
    #     mask = y == label
    #     plt.scatter(X[mask, 0], X[mask, 1], c=color, marker=marker,
    #                s=150, label=f'Class {label}', edgecolors='black')
    
    # TODO: Plot decision boundary
    # w1, w2 = perceptron.weights
    # b = perceptron.bias
    # if abs(w2) > 1e-6:
    #     x1_range = np.linspace(-0.5, 1.5, 100)
    #     x2_boundary = -(w1 * x1_range + b) / w2
    #     plt.plot(x1_range, x2_boundary, 'b-', linewidth=2, label='Decision Boundary')
    
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# =============================================================================
# MAIN: Test Your Implementation
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PERCEPTRON FROM SCRATCH")
    print("=" * 60)
    
    # AND Gate
    print("\n--- AND GATE ---")
    X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_and = np.array([0, 0, 0, 1])
    
    # TODO: Create and train perceptron
    # perceptron_and = Perceptron(n_features=2, learning_rate=0.1)
    # perceptron_and, history_and = perceptron_and.fit(X_and, y_and)
    # 
    # predictions = perceptron_and.predict(X_and)
    # print(f"Predictions: {predictions}")
    # print(f"Actual: {y_and}")
    # print(f"Accuracy: {(predictions == y_and).mean():.2%}")
    # 
    # plot_decision_boundary(perceptron_and, X_and, y_and, 'AND Gate')
    
    # OR Gate
    print("\n--- OR GATE ---")
    X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_or = np.array([0, 1, 1, 1])
    
    # TODO: Create and train perceptron for OR
    
    # XOR Gate
    print("\n--- XOR GATE (The Challenge!) ---")
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([0, 1, 1, 0])
    
    # TODO: Attempt XOR
    # perceptron_xor = Perceptron(n_features=2, learning_rate=0.1)
    # perceptron_xor, history_xor = perceptron_xor.fit(X_xor, y_xor, epochs=1000)
    
    # =============================================================================
    # REFLECTION QUESTIONS
    # =============================================================================
    
    # Q1: Did the perceptron converge for XOR?
    # Answer:
    
    # Q2: Why can't a single perceptron learn XOR?
    # Answer:
    
    # Q3: What would you need to solve XOR?
    # Answer:

