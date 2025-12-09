"""
Exercise 01: Linear Regression from Scratch - SOLUTION
=======================================================

Complete implementation of gradient descent for linear regression.
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# PART 1: DATA AND BUILDING BLOCKS
# =============================================================================

# Generate training data
np.random.seed(42)
X = np.random.uniform(0, 10, 100)
y = 2.5 * X + 7 + np.random.normal(0, 2, 100)

# Visualize the data
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.6, edgecolors='black', linewidth=0.5)
plt.title('Training Data')
plt.xlabel('X')
plt.ylabel('y')
plt.grid(True, alpha=0.3)
plt.show()


def predict(X, w, b):
    """Compute predictions using linear equation."""
    return w * X + b


def compute_mse(y_true, y_pred):
    """Compute Mean Squared Error."""
    n = len(y_true)
    return (1/n) * np.sum((y_pred - y_true) ** 2)


def compute_gradients(X, y_true, y_pred):
    """Compute gradients for w and b."""
    n = len(X)
    error = y_pred - y_true
    dw = (2/n) * np.sum(error * X)
    db = (2/n) * np.sum(error)
    return dw, db


# =============================================================================
# PART 2: TRAINING LOOP
# =============================================================================

def gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    """Train linear regression using gradient descent."""
    # Initialize weights
    w = np.random.randn() * 0.01
    b = 0.0
    
    history = {'loss': [], 'w': [], 'b': []}
    
    for epoch in range(epochs):
        # Step 1: Make predictions
        y_pred = predict(X, w, b)
        
        # Step 2: Compute loss
        loss = compute_mse(y, y_pred)
        
        # Step 3: Compute gradients
        dw, db = compute_gradients(X, y, y_pred)
        
        # Step 4: Update weights
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        # Record history
        history['loss'].append(loss)
        history['w'].append(w)
        history['b'].append(b)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}, w = {w:.4f}, b = {b:.4f}")
    
    return w, b, history


# =============================================================================
# PART 3: TRAIN AND VISUALIZE
# =============================================================================

if __name__ == "__main__":
    print("Training Linear Regression from Scratch...")
    print("=" * 50)
    
    w_final, b_final, history = gradient_descent(X, y, learning_rate=0.01, epochs=1000)
    
    print("\n" + "=" * 50)
    print(f"Final Model: y = {w_final:.4f} * x + {b_final:.4f}")
    print(f"True Relationship: y = 2.5 * x + 7")
    
    # Plot the loss curve
    plt.figure(figsize=(10, 4))
    plt.plot(history['loss'], 'b-', linewidth=2)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Plot the final fit
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.6, edgecolors='black', linewidth=0.5, label='Data')
    
    # Regression line
    X_line = np.linspace(0, 10, 100)
    y_line = predict(X_line, w_final, b_final)
    plt.plot(X_line, y_line, 'r-', linewidth=2, label=f'Learned: y = {w_final:.2f}x + {b_final:.2f}')
    
    # True line
    y_true_line = 2.5 * X_line + 7
    plt.plot(X_line, y_true_line, 'g--', linewidth=2, label='True: y = 2.5x + 7')
    
    plt.title('Linear Regression from Scratch')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Weight evolution animation
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['w'], label='w')
    plt.axhline(y=2.5, color='g', linestyle='--', label='True w=2.5')
    plt.xlabel('Epoch')
    plt.ylabel('Weight')
    plt.title('Weight Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['b'], label='b')
    plt.axhline(y=7, color='g', linestyle='--', label='True b=7')
    plt.xlabel('Epoch')
    plt.ylabel('Bias')
    plt.title('Bias Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # =============================================================================
    # PART 4: COMPARE WITH SKLEARN
    # =============================================================================
    
    from sklearn.linear_model import LinearRegression
    
    sklearn_model = LinearRegression()
    sklearn_model.fit(X.reshape(-1, 1), y)
    
    print("\n" + "=" * 50)
    print("COMPARISON WITH SKLEARN")
    print("=" * 50)
    print(f"Your Model:    w = {w_final:.4f}, b = {b_final:.4f}")
    print(f"sklearn Model: w = {sklearn_model.coef_[0]:.4f}, b = {sklearn_model.intercept_:.4f}")
    print(f"True Values:   w = 2.5000, b = 7.0000")
    
    # =============================================================================
    # REFLECTION ANSWERS
    # =============================================================================
    
    # Q1: What happens with learning_rate = 0.1? learning_rate = 1.0?
    # Answer: With lr=0.1, training converges faster but may overshoot.
    #         With lr=1.0, the loss explodes (diverges) because steps are too large.
    #         The gradients cause weights to oscillate wildly and never converge.
    
    # Q2: What happens with learning_rate = 0.0001?
    # Answer: Training becomes very slow. After 1000 epochs, the model hasn't
    #         converged yet. You'd need many more epochs (10,000+) to reach
    #         similar accuracy. Trade-off between speed and stability.
    
    # Q3: Why small random initialization instead of zeros?
    # Answer: For linear regression, zeros would work fine. But this habit
    #         prepares us for neural networks where zero initialization causes
    #         all neurons to learn the same thing (symmetry problem).
    #         Small random values break symmetry.
    
    # Q4: How close did you get to true values (w=2.5, b=7)?
    # Answer: Very close! w is typically within 0.1 of 2.5, b within 0.5 of 7.
    #         The noise in data prevents perfect recovery. sklearn uses
    #         closed-form solution (Normal Equation) which is more precise
    #         than gradient descent for this simple case.

