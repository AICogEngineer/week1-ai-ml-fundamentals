# Regression Logic and Applications

## Learning Objectives

- Understand regression as predicting continuous numerical values
- Explain the linear regression equation and interpret its components
- Describe the cost function concept and its role in learning
- Identify real-world applications of regression algorithms

## Why This Matters

Regression is everywhere - and you've been consuming its predictions your whole life without realizing it.

When your weather app predicts "high of 75 degrees," when a real estate site estimates a home's value, when your fitness tracker projects calories burned, when an economist forecasts GDP growth - these are all regression models at work.

In our **From Zero to Neural** journey, regression represents the purest form of "learning from data." Master this, and you'll understand the foundation that deep learning builds upon: finding mathematical relationships in data.

## The Concept

### What Is Regression?

**Regression** is a supervised learning technique for predicting **continuous numerical values**.

Key distinction:
- **Classification**: Predicts categories (spam/not spam, cat/dog)
- **Regression**: Predicts numbers on a continuous scale (price, temperature, age)

```
Classification Output:  "Cat" or "Dog"  (discrete)
Regression Output:      $247,500        (continuous)
```

### Linear Regression: The Foundation

Linear regression finds the best straight line through your data points.

**The Equation:**

```
y = mx + b

Where:
  y = predicted value (target/dependent variable)
  x = input feature (independent variable)
  m = slope (how much y changes per unit of x)
  b = intercept (y value when x = 0)
```

In machine learning notation:

```
y = w * x + b

Where:
  w = weight (same as slope)
  b = bias (same as intercept)
```

**Visual Intuition:**

```
Price ($)
    ^
400k|                          *
    |                    *   /
300k|              *   /
    |        *   /
200k|    * /
    |  /
100k|/
    +-------------------------> Square Feet
       1000  1500  2000  2500

The line represents: Price = w * SquareFeet + b
```

### Multiple Linear Regression

Real-world predictions rarely depend on a single feature. Multiple linear regression handles multiple inputs:

```
y = w1*x1 + w2*x2 + w3*x3 + ... + b

House Price Example:
price = (w1 * sqft) + (w2 * bedrooms) + (w3 * age) + b
```

Each weight (w) indicates how much that feature contributes to the prediction.

### The Cost Function: How the Model Learns

The model needs a way to measure "how wrong" its predictions are. This is the **cost function** (also called loss function).

**Mean Squared Error (MSE)** - The most common regression cost function:

```
MSE = (1/n) * SUM[(actual - predicted)^2]
```

Why squared?
1. Makes all errors positive (no cancellation)
2. Penalizes large errors more than small errors
3. Mathematically convenient (differentiable)

**Example Calculation:**

| Actual Price | Predicted Price | Error | Squared Error |
|--------------|-----------------|-------|---------------|
| $250,000 | $240,000 | $10,000 | 100,000,000 |
| $300,000 | $320,000 | -$20,000 | 400,000,000 |
| $200,000 | $195,000 | $5,000 | 25,000,000 |

```
MSE = (100M + 400M + 25M) / 3 = $175,000,000

RMSE (Root MSE) = sqrt(175M) = ~$13,228
(More interpretable - in same units as target)
```

### The Learning Process

Linear regression "learns" by finding the weights that minimize the cost function:

```
1. Start with random weights (w, b)
2. Make predictions: y_pred = w*x + b
3. Calculate error: MSE(y_actual, y_pred)
4. Adjust weights to reduce error
5. Repeat until error stops decreasing
```

This process of adjusting weights is called **optimization**. The most common method is **gradient descent**, which we'll explore in depth during Week 2.

### Beyond Linearity: Polynomial Regression

What if the relationship isn't linear?

```
Salary vs. Experience (not linear):

Salary
   ^
80k|                    ****
   |              ****
60k|         ***
   |      **
40k|   **
   | *
   +-------------------------> Years
     0   5   10   15   20
```

Polynomial regression adds powers of x:

```
y = w1*x + w2*x^2 + w3*x^3 + b
```

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Create polynomial features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Fit linear regression on polynomial features
model = LinearRegression()
model.fit(X_poly, y)
```

## Code Example: Complete Regression Workflow

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Sample data: House features and prices
# Features: [square_feet, bedrooms, bathrooms, age]
X = np.array([
    [1500, 3, 2, 10],
    [2000, 4, 2, 5],
    [1200, 2, 1, 20],
    [1800, 3, 2, 8],
    [2200, 4, 3, 3],
    [1600, 3, 2, 15],
    [1900, 3, 2, 7],
    [2400, 5, 3, 2],
])

# Target: House prices
y = np.array([250000, 350000, 180000, 280000, 400000, 260000, 310000, 450000])

# Split data: 75% training, 25% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Examine learned parameters
print("Learned weights:")
print(f"  Square Feet: ${model.coef_[0]:.2f} per sq ft")
print(f"  Bedrooms:    ${model.coef_[1]:.2f} per bedroom")
print(f"  Bathrooms:   ${model.coef_[2]:.2f} per bathroom")
print(f"  Age:         ${model.coef_[3]:.2f} per year")
print(f"  Bias:        ${model.intercept_:.2f}")

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"  RMSE: ${rmse:,.2f}")
print(f"  R-squared: {r2:.3f}")

# Predict price for a new house
new_house = np.array([[1750, 3, 2, 12]])
predicted_price = model.predict(new_house)
print(f"\nPredicted price for new house: ${predicted_price[0]:,.2f}")
```

**Sample Output:**
```
Learned weights:
  Square Feet: $121.45 per sq ft
  Bedrooms:    $15234.12 per bedroom
  Bathrooms:   $8921.33 per bathroom
  Age:         $-2156.78 per year
  Bias:        $-12453.21

Model Performance:
  RMSE: $18,234.56
  R-squared: 0.945

Predicted price for new house: $267,890.00
```

### Interpreting the Results

- **Coefficients (weights)** show feature importance:
  - Each square foot adds ~$121 to price
  - Each bedroom adds ~$15,234
  - Each year of age *subtracts* ~$2,157 (negative weight)

- **R-squared (R2)**: Proportion of variance explained
  - 0.945 means the model explains 94.5% of price variation
  - Range: 0 (worst) to 1 (perfect)

- **RMSE**: Average prediction error in original units
  - $18,234 means predictions are typically off by this amount

## Real-World Applications

| Domain | Regression Application | Target Variable |
|--------|----------------------|-----------------|
| Real Estate | Price estimation | Sale price |
| Finance | Stock price prediction | Future price |
| Healthcare | Disease progression | Severity score |
| Marketing | Customer lifetime value | Revenue |
| Weather | Temperature forecasting | Degrees |
| E-commerce | Demand forecasting | Units sold |
| HR | Salary prediction | Compensation |
| Energy | Load forecasting | Power consumption |

### Industry Example: Insurance Pricing

Insurance companies use regression to set premiums:

```python
# Features that predict insurance claims
features = [
    'age',
    'bmi',
    'smoker',
    'region',
    'children'
]

# Model predicts: Expected annual claims cost
# Premium = predicted_claims + profit_margin
```

## Common Pitfalls

1. **Extrapolation**: Don't predict outside your training data range
   - Model trained on houses 1000-3000 sqft shouldn't predict for 10,000 sqft

2. **Multicollinearity**: Correlated features cause unstable weights
   - If `bedrooms` and `sqft` are highly correlated, individual weights are unreliable

3. **Outliers**: Single extreme values can skew the entire line
   - One $10M mansion in a dataset of $200K homes distorts everything

4. **Assuming Linearity**: Not all relationships are linear
   - Check residual plots; consider polynomial features if needed

## Key Takeaways

1. **Regression predicts continuous values** - numbers on a scale, not categories.

2. **Linear regression finds the best-fit line** using the equation `y = wx + b`.

3. **The cost function measures error** - MSE penalizes predictions far from actual values.

4. **Weights reveal feature importance** - larger absolute weights = more influence.

5. **R-squared and RMSE evaluate performance** - how well does the model explain variance and how far off are predictions?

## Looking Ahead

Regression handles continuous predictions, but what about categorical outcomes? The next reading on **Classification** shows how we adapt these concepts when the target is discrete categories rather than continuous values.

Later this week, you'll see how neural networks can perform regression tasks - but with the power to learn complex, non-linear relationships that simple linear models can't capture.

## Additional Resources

- [Linear Regression - StatQuest](https://www.youtube.com/watch?v=nk2CQITm_eo) - Visual explanation with clear math
- [scikit-learn Linear Regression](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares) - Official documentation
- [Regression Metrics Guide](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics) - Understanding MSE, MAE, R2

