"""
Exercise 01: Linear Regression from Scratch
============================================

Complete the TODO sections to implement gradient descent.
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# PART 1: DATA AND BUILDING BLOCKS
# =============================================================================

# Generate training data
np.random.seed(42)
X = np.random.uniform(0, 10, 100)  # 100 points between 0 and 10
y = 2.5 * X + 7 + np.random.normal(0, 2, 100)  # True: y = 2.5x + 7 + noise

# TODO: Visualize the data
# plt.scatter(...)
# plt.title('Training Data')
# plt.xlabel('X')
# plt.ylabel('y')
# plt.show()


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


# =============================================================================
# PART 2: TRAINING LOOP
# =============================================================================

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
    # Initialize weights
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
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}, w = {w:.4f}, b = {b:.4f}")
    
    return w, b, history


# =============================================================================
# PART 3: TRAIN AND VISUALIZE
# =============================================================================

if __name__ == "__main__":
    # Train the model
    print("Training Linear Regression from Scratch...")
    print("=" * 50)
    
    w_final, b_final, history = gradient_descent(X, y, learning_rate=0.01, epochs=1000)
    
    print("\n" + "=" * 50)
    print(f"Final Model: y = {w_final:.4f} * x + {b_final:.4f}")
    print(f"True Relationship: y = 2.5 * x + 7")
    
    # TODO: Plot the loss curve
    # plt.figure(figsize=(10, 4))
    # ...
    
    # TODO: Plot the final fit
    # plt.figure(figsize=(10, 6))
    # ...
    
    # =============================================================================
    # PART 4: COMPARE WITH SKLEARN
    # =============================================================================
    
    from sklearn.linear_model import LinearRegression
    
    # TODO: Train sklearn model
    # sklearn_model = LinearRegression()
    # sklearn_model.fit(...)
    
    # TODO: Compare weights
    # print(f"Your Model:    w = {w_final:.4f}, b = {b_final:.4f}")
    # print(f"sklearn Model: w = {sklearn_model.coef_[0]:.4f}, b = {sklearn_model.intercept_:.4f}")
    
    # =============================================================================
    # REFLECTION QUESTIONS (Answer in comments below)
    # =============================================================================
    
    # Q1: What happens with learning_rate = 0.1? learning_rate = 1.0?
    # Answer: 
    
    # Q2: What happens with learning_rate = 0.0001?
    # Answer: 
    
    # Q3: Why small random initialization instead of zeros?
    # Answer: 
    
    # Q4: How close did you get to true values (w=2.5, b=7)?
    # Answer: 

