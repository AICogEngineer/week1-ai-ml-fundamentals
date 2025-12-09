"""
Exercise 06: Forward Propagation Calculator
==========================================

Verify your hand calculations with NumPy.
"""

import numpy as np

# =============================================================================
# GIVEN NETWORK PARAMETERS
# =============================================================================

# Hidden Layer weights and bias
W1 = np.array([[0.2, -0.5],
               [0.3,  0.4],
               [-0.1, 0.2]])
b1 = np.array([0.1, -0.1])

# Output Layer weights and bias
W2 = np.array([[0.6],
               [-0.3]])
b2 = np.array([0.1])

# Input
X = np.array([1.0, 0.5, -0.5])


# =============================================================================
# ACTIVATION FUNCTIONS
# =============================================================================

def relu(z):
    """ReLU activation."""
    return np.maximum(0, z)


def sigmoid(z):
    """Sigmoid activation."""
    return 1 / (1 + np.exp(-z))


# =============================================================================
# FORWARD PROPAGATION
# =============================================================================

print("=" * 60)
print("FORWARD PROPAGATION CALCULATOR")
print("=" * 60)

print(f"\nInput X = {X}")
print(f"W1 shape: {W1.shape}, b1 shape: {b1.shape}")
print(f"W2 shape: {W2.shape}, b2 shape: {b2.shape}")

# TODO: Step 1 - Hidden Layer Linear Transform
# z1 = X @ W1 + b1
z1 = None  # Your code

print(f"\n--- Step 1: z1 = X @ W1 + b1 ---")
# print(f"z1 = {z1}")

# TODO: Step 2 - Hidden Layer Activation (ReLU)
# a1 = relu(z1)
a1 = None  # Your code

print(f"\n--- Step 2: a1 = ReLU(z1) ---")
# print(f"a1 = {a1}")

# TODO: Step 3 - Output Layer Linear Transform
# z2 = a1 @ W2 + b2
z2 = None  # Your code

print(f"\n--- Step 3: z2 = a1 @ W2 + b2 ---")
# print(f"z2 = {z2}")

# TODO: Step 4 - Output Layer Activation (Sigmoid)
# y_hat = sigmoid(z2)
y_hat = None  # Your code

print(f"\n--- Step 4: y_hat = sigmoid(z2) ---")
# print(f"y_hat = {y_hat}")

# TODO: Interpret the output
print(f"\n--- Interpretation ---")
# print(f"Probability of positive class: {y_hat[0]:.4f}")
# print(f"Classification (threshold=0.5): {int(y_hat[0] >= 0.5)}")


# =============================================================================
# PART 3: DIFFERENT INPUTS
# =============================================================================

def forward_pass_simple(X, W1, b1, W2, b2):
    """Complete forward pass for our 2-layer network."""
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    y_hat = sigmoid(z2)
    return z1, a1, z2, y_hat


print("\n" + "=" * 60)
print("TESTING DIFFERENT INPUTS")
print("=" * 60)

inputs = [
    np.array([0.0, 1.0, 1.0]),    # Input 2
    np.array([-1.0, 0.0, 0.0]),   # Input 3
    np.array([2.0, 2.0, 2.0])     # Input 4
]

for i, X_test in enumerate(inputs, start=2):
    print(f"\n--- Input {i}: X = {X_test} ---")
    # TODO: Run forward pass and print results
    # z1, a1, z2, y_hat = forward_pass_simple(X_test, W1, b1, W2, b2)
    # print(f"z1 = {z1}")
    # print(f"a1 = {a1}")
    # print(f"z2 = {z2}")
    # print(f"y_hat = {y_hat[0]:.4f}")


# =============================================================================
# ANALYSIS QUESTIONS
# =============================================================================

# Q1: Which input(s) produced y_hat > 0.5?
# Answer:

# Q2: For Input 3, what happened at ReLU? Why?
# Answer:

# Q3: How does the network's output change with different inputs?
# Answer:

