"""
Exercise 05: Activation Function Analysis
=========================================

Implement and analyze activation functions.
"""

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# PART 1: IMPLEMENT ACTIVATION FUNCTIONS
# =============================================================================

def step(z):
    """Step function (Heaviside)."""
    # TODO: Return 1 if z >= 0, else 0
    pass


def sigmoid(z):
    """Sigmoid / Logistic function."""
    # TODO: Return 1 / (1 + exp(-z))
    # Hint: Use np.clip(z, -500, 500) to avoid overflow
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


# =============================================================================
# PART 2: IMPLEMENT DERIVATIVES
# =============================================================================

def sigmoid_derivative(z):
    """Derivative of sigmoid."""
    # TODO: sigmoid(z) * (1 - sigmoid(z))
    pass


def tanh_derivative(z):
    """Derivative of tanh."""
    # TODO: 1 - tanh(z)^2
    pass


def relu_derivative(z):
    """Derivative of ReLU."""
    # TODO: 1 if z > 0, else 0
    pass


def leaky_relu_derivative(z, alpha=0.01):
    """Derivative of Leaky ReLU."""
    # TODO: 1 if z > 0, else alpha
    pass


# =============================================================================
# PART 3: VISUALIZATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ACTIVATION FUNCTION ANALYSIS")
    print("=" * 60)
    
    z = np.linspace(-5, 5, 1000)
    
    # TODO: Create subplot grid for activations
    # fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    # 
    # activations = [
    #     ('Step', step),
    #     ('Sigmoid', sigmoid),
    #     ('Tanh', tanh_activation),
    #     ('ReLU', relu),
    #     ('Leaky ReLU', leaky_relu)
    # ]
    # 
    # for idx, (name, func) in enumerate(activations):
    #     ax = axes[idx // 3, idx % 3]
    #     ax.plot(z, func(z), linewidth=2)
    #     ax.axhline(y=0, color='black', linewidth=0.5)
    #     ax.axvline(x=0, color='black', linewidth=0.5)
    #     ax.set_title(name)
    #     ax.grid(True, alpha=0.3)
    # 
    # plt.tight_layout()
    # plt.show()
    
    # =============================================================================
    # EXPERIMENT A: OUTPUT RANGES
    # =============================================================================
    
    print("\n--- OUTPUT RANGES ---")
    # TODO: Fill in the table
    # | Activation | Min Output | Max Output | Zero-Centered? |
    
    # =============================================================================
    # EXPERIMENT B: GRADIENT VALUES
    # =============================================================================
    
    print("\n--- GRADIENT VALUES ---")
    test_points = [-3, -1, 0, 1, 3]
    
    # TODO: Calculate gradient values at test points
    
    # =============================================================================
    # EXPERIMENT C: VANISHING GRADIENT
    # =============================================================================
    
    print("\n--- VANISHING GRADIENT ---")
    # TODO: What is sigmoid gradient at z = 10? z = -10?
    
    # =============================================================================
    # EXPERIMENT D: DEAD RELU
    # =============================================================================
    
    print("\n--- DEAD RELU ---")
    # TODO: Explain the dying ReLU problem

