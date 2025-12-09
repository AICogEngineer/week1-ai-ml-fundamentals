"""
Demo 01: Linear Regression Playground
=====================================
Week 1, Tuesday - AI/ML Fundamentals

This demo introduces supervised learning through linear regression.
We predict house prices from square footage - a classic ML example.

INSTRUCTOR NOTES:
- Run cells incrementally to show the learning process
- Pause after the scatter plot to discuss the "best fit line" concept
- Emphasize: the model LEARNS the relationship from data

Estimated Time: 15-20 minutes
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# =============================================================================
# SECTION 1: THE DATA
# =============================================================================
print("=" * 60)
print("DEMO 01: LINEAR REGRESSION PLAYGROUND")
print("Predicting House Prices from Square Footage")
print("=" * 60)

# Generate synthetic house data
# In real ML, this would come from a database or CSV file
np.random.seed(42)  # For reproducibility

# Features: Square footage (1000 to 3500 sq ft)
square_feet = np.random.uniform(1000, 3500, 100)

# Target: Price follows a linear relationship + some noise
# Price = $100 per sqft + $50,000 base + random variation
price = 100 * square_feet + 50000 + np.random.normal(0, 20000, 100)

print("\n--- THE DATA ---")
print(f"Number of houses: {len(square_feet)}")
print(f"Square footage range: {square_feet.min():.0f} - {square_feet.max():.0f}")
print(f"Price range: ${price.min():,.0f} - ${price.max():,.0f}")

# Show first few examples
print("\nFirst 5 houses:")
print(f"{'Sq Ft':<10} {'Price':<15}")
print("-" * 25)
for i in range(5):
    print(f"{square_feet[i]:<10.0f} ${price[i]:,.0f}")

# =============================================================================
# SECTION 2: VISUALIZE THE RAW DATA
# =============================================================================
# INSTRUCTOR: Pause here and ask "What pattern do you see?"

plt.figure(figsize=(10, 6))
plt.scatter(square_feet, price, alpha=0.6, edgecolors='black', linewidth=0.5)
plt.xlabel('Square Footage', fontsize=12)
plt.ylabel('Price ($)', fontsize=12)
plt.title('House Prices vs Square Footage\n(Before Machine Learning)', fontsize=14)
plt.grid(True, alpha=0.3)

# Format y-axis as currency
ax = plt.gca()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

plt.tight_layout()
plt.savefig('01_raw_data.png', dpi=100)
plt.show()

print("\n[Saved: 01_raw_data.png]")
print("DISCUSSION: What relationship do you see between size and price?")

# =============================================================================
# SECTION 3: PREPARE DATA FOR MACHINE LEARNING
# =============================================================================
print("\n--- PREPARING DATA ---")

# Reshape for sklearn (needs 2D array for features)
X = square_feet.reshape(-1, 1)  # (100, 1) - 100 samples, 1 feature
y = price                        # (100,) - 100 target values

print(f"Features shape (X): {X.shape}")
print(f"Target shape (y): {y.shape}")

# Split into training and testing sets
# INSTRUCTOR: Explain why we need separate train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {len(X_train)} (80%)")
print(f"Testing samples: {len(X_test)} (20%)")
print("\nWhy split? We train on 80%, then test on the 20% the model has NEVER seen.")

# =============================================================================
# SECTION 4: TRAIN THE MODEL
# =============================================================================
print("\n--- TRAINING THE MODEL ---")

# Create the model
model = LinearRegression()

# Fit the model to training data
# THIS IS WHERE THE LEARNING HAPPENS
print("Fitting model to training data...")
model.fit(X_train, y_train)
print("Training complete!")

# Examine what the model learned
print(f"\nLearned Parameters:")
print(f"  Slope (per sq ft): ${model.coef_[0]:,.2f}")
print(f"  Intercept (base price): ${model.intercept_:,.2f}")
print(f"\nThe model learned: Price = ${model.coef_[0]:.2f} x SqFt + ${model.intercept_:,.2f}")

# =============================================================================
# SECTION 5: VISUALIZE THE LEARNED MODEL
# =============================================================================
print("\n--- VISUALIZING THE MODEL ---")

# Create prediction line
X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_line = model.predict(X_line)

plt.figure(figsize=(10, 6))

# Plot training data
plt.scatter(X_train, y_train, alpha=0.6, label='Training Data', 
            edgecolors='black', linewidth=0.5, c='blue')

# Plot test data
plt.scatter(X_test, y_test, alpha=0.6, label='Test Data', 
            edgecolors='black', linewidth=0.5, c='green', marker='s')

# Plot the regression line
plt.plot(X_line, y_line, 'r-', linewidth=2, label='Learned Model')

plt.xlabel('Square Footage', fontsize=12)
plt.ylabel('Price ($)', fontsize=12)
plt.title('Linear Regression: The Model Has Learned!\n'
          f'Price = ${model.coef_[0]:.0f} x SqFt + ${model.intercept_:,.0f}', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

ax = plt.gca()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

plt.tight_layout()
plt.savefig('01_regression_line.png', dpi=100)
plt.show()

print("[Saved: 01_regression_line.png]")

# =============================================================================
# SECTION 6: MAKE PREDICTIONS
# =============================================================================
print("\n--- MAKING PREDICTIONS ---")

# Predict on test set
y_pred = model.predict(X_test)

# Show some predictions vs actual
print("Sample Predictions vs Actual:")
print(f"{'Sq Ft':<10} {'Predicted':<15} {'Actual':<15} {'Error':<15}")
print("-" * 55)
for i in range(5):
    error = y_pred[i] - y_test[i]
    print(f"{X_test[i][0]:<10.0f} ${y_pred[i]:>12,.0f} ${y_test[i]:>12,.0f} ${error:>+12,.0f}")

# =============================================================================
# SECTION 7: EVALUATE THE MODEL
# =============================================================================
print("\n--- MODEL EVALUATION ---")

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:,.0f}")
print(f"Root MSE (RMSE): ${rmse:,.0f}")
print(f"R-squared Score: {r2:.4f}")

print(f"\nInterpretation:")
print(f"  - RMSE: Predictions are typically off by ~${rmse:,.0f}")
print(f"  - R-squared: Model explains {r2*100:.1f}% of price variation")

# =============================================================================
# SECTION 8: INTERACTIVE PREDICTION
# =============================================================================
print("\n--- TRY YOUR OWN PREDICTION ---")

# Predict for specific square footage
test_sqft = [1500, 2000, 2500, 3000]

print("Predictions for various house sizes:")
print(f"{'Square Feet':<15} {'Predicted Price':<20}")
print("-" * 35)
for sqft in test_sqft:
    pred = model.predict([[sqft]])[0]
    print(f"{sqft:<15} ${pred:>15,.0f}")

# =============================================================================
# SECTION 9: KEY TAKEAWAYS
# =============================================================================
print("\n" + "=" * 60)
print("KEY TAKEAWAYS")
print("=" * 60)
print("""
1. SUPERVISED LEARNING: We provided features (X) AND labels (y)
   The model learned the relationship between them.

2. LINEAR REGRESSION: Finds the best-fit line through the data
   Equation: y = slope * x + intercept

3. TRAIN/TEST SPLIT: Always evaluate on data the model hasn't seen
   This tells us how well it will work on NEW houses.

4. METRICS:
   - RMSE tells us average prediction error (in dollars)
   - R-squared tells us how much variation we explain (0-1)

5. THE MODEL LEARNED: 
   Each square foot adds ~${:.0f} to the price
   Base price is ~${:,.0f}
""".format(model.coef_[0], model.intercept_))

print("Next: We'll see CLASSIFICATION (predicting categories, not numbers)")

