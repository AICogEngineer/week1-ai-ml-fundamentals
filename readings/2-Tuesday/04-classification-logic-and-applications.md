# Classification Logic and Applications

## Learning Objectives

- Understand classification as predicting discrete categories
- Explain decision boundaries and their role in separating classes
- Develop intuition for logistic regression as a classification algorithm
- Identify real-world applications of classification across industries

## Why This Matters

Every time you unlock your phone with facial recognition, every time Gmail filters a spam email, every time a doctor uses a diagnostic tool to assess disease risk - a classification algorithm is making a decision.

Classification is arguably the most widely deployed form of machine learning. In our **From Zero to Neural** journey, understanding how machines make categorical decisions is essential before we build neural networks that classify images, text, and more.

The stakes are high: a misclassified spam email is annoying; a misclassified tumor is life-threatening. Understanding classification means understanding how to build systems that make reliable decisions.

## The Concept

### What Is Classification?

**Classification** is a supervised learning technique for predicting **discrete categorical outcomes**.

Unlike regression (continuous values), classification outputs belong to predefined categories:

```
Regression Output:   $247,500.00  (any value on continuous scale)
Classification Output: "Approved" or "Denied"  (discrete categories)
```

### Binary vs. Multi-Class Classification

**Binary Classification**: Two possible outcomes
- Spam / Not Spam
- Fraud / Legitimate
- Click / No Click
- Positive / Negative

**Multi-Class Classification**: More than two outcomes
- Dog / Cat / Bird / Fish
- Low Risk / Medium Risk / High Risk
- Sentiment: Positive / Neutral / Negative

```python
# Binary classification
labels = ['spam', 'not_spam']

# Multi-class classification
labels = ['setosa', 'versicolor', 'virginica']
```

### Decision Boundaries

The core job of a classifier is to draw **decision boundaries** - dividing lines (or surfaces) that separate different classes.

**Visual Intuition (2D):**

```
Feature 2 (Height)
      ^
  6'2"|      o  o      |  Class A (o)
      |    o    o      |  Class B (x)
  5'8"|  ------+-------   <-- Decision Boundary
      |    x   |x  x
  5'4"|  x   x |  x
      +-------------------> Feature 1 (Weight)
         140  160  180
```

Everything above the line is classified as Class A; below is Class B.

**Key insight**: Classification algorithms learn where to place these boundaries based on training data.

### Logistic Regression: Classification Despite the Name

Despite "regression" in its name, **logistic regression** is a classification algorithm. It predicts the *probability* of belonging to a class.

**The Logistic (Sigmoid) Function:**

```
P(y=1) = 1 / (1 + e^(-z))

Where: z = w*x + b (linear combination)
```

The sigmoid function squashes any input to a value between 0 and 1:

```
Probability
    1.0 |           *******
        |         **
    0.5 |--------*---------
        |      **
    0.0 |******
        +--------------------> z (linear combination)
          -6  -3   0   3   6
```

**Making Decisions:**

```
If P(y=1) >= 0.5: Predict Class 1
If P(y=1) < 0.5:  Predict Class 0
```

The 0.5 threshold creates the decision boundary.

### Why Not Use Linear Regression for Classification?

Linear regression outputs any real number, but classification needs bounded probabilities:

```
Linear Regression Problem:

Probability
    2.0 |               *
    1.0 |           *       <-- Impossible! Probability > 1
    0.5 |       *
    0.0 |   *
   -0.5 |*                  <-- Impossible! Probability < 0
        +----------------------> Input
```

Logistic regression's sigmoid function guarantees outputs between 0 and 1.

### Beyond Logistic Regression: Other Classifiers

Logistic regression is just one approach. Others include:

| Algorithm | Decision Boundary | Strengths |
|-----------|------------------|-----------|
| Logistic Regression | Linear | Interpretable, fast, probabilistic |
| K-Nearest Neighbors | Non-linear | Simple, no training phase |
| Decision Trees | Axis-aligned | Interpretable, handles non-linear |
| Support Vector Machines | Linear/Non-linear | Effective in high dimensions |
| Random Forests | Complex | Robust, handles noise |
| Neural Networks | Any shape | Learns complex patterns |

```
Different Decision Boundaries:

Logistic Regression:     Decision Tree:        KNN:
      |                       |                 .***.
   o o|x x                o o |x x            .* x x*.
   o o|x x                ----+----           * o o | *
   o  |x x x              o o |x x x          *.o o.|*
      |                       |                 *...*
  (linear)              (rectangular)         (organic)
```

## Code Example: Complete Classification Workflow

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Sample data: Customer features for churn prediction
# Features: [tenure_months, monthly_charges, total_charges, support_tickets]
X = np.array([
    [12, 70, 840, 2],    # Stayed
    [24, 80, 1920, 1],   # Stayed
    [3, 90, 270, 5],     # Churned
    [36, 75, 2700, 0],   # Stayed
    [6, 95, 570, 4],     # Churned
    [48, 65, 3120, 1],   # Stayed
    [2, 85, 170, 6],     # Churned
    [60, 70, 4200, 0],   # Stayed
    [4, 100, 400, 3],    # Churned
    [18, 72, 1296, 2],   # Stayed
])

# Target: 0 = Stayed, 1 = Churned
y = np.array([0, 0, 1, 0, 1, 0, 1, 0, 1, 0])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Scale features (important for logistic regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression classifier
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)

# Evaluate
print("Predictions vs Actual:")
for i, (pred, actual, prob) in enumerate(zip(y_pred, y_test, y_prob)):
    status = "Correct" if pred == actual else "Wrong"
    label = "Churn" if pred == 1 else "Stay"
    print(f"  Sample {i+1}: Predicted {label} (prob: {prob[1]:.2f}) - {status}")

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2%}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['Stay', 'Churn'])}")
```

**Sample Output:**
```
Predictions vs Actual:
  Sample 1: Predicted Stay (prob: 0.23) - Correct
  Sample 2: Predicted Churn (prob: 0.78) - Correct
  Sample 3: Predicted Stay (prob: 0.15) - Correct

Accuracy: 100.00%

Confusion Matrix:
[[2 0]
 [0 1]]

Classification Report:
              precision    recall  f1-score   support
        Stay       1.00      1.00      1.00         2
       Churn       1.00      1.00      1.00         1
    accuracy                           1.00         3
```

### Understanding the Confusion Matrix

```
                  Predicted
                Stay    Churn
Actual  Stay    [TN]    [FP]     True Negative, False Positive
        Churn   [FN]    [TP]     False Negative, True Positive
```

| Metric | Formula | Meaning |
|--------|---------|---------|
| Accuracy | (TP+TN) / Total | Overall correctness |
| Precision | TP / (TP+FP) | Of predicted positives, how many were correct? |
| Recall | TP / (TP+FN) | Of actual positives, how many did we find? |
| F1-Score | 2 * (Prec * Rec) / (Prec + Rec) | Harmonic mean of precision and recall |

**When metrics matter differently:**
- **High precision needed**: Spam detection (don't mark legitimate emails as spam)
- **High recall needed**: Disease screening (don't miss sick patients)

## Real-World Applications

| Domain | Classification Task | Classes |
|--------|---------------------|---------|
| Email | Spam filtering | Spam / Not Spam |
| Healthcare | Disease diagnosis | Positive / Negative |
| Finance | Fraud detection | Fraud / Legitimate |
| E-commerce | Purchase prediction | Buy / Not Buy |
| Security | Intrusion detection | Attack / Normal |
| HR | Resume screening | Qualified / Not Qualified |
| Manufacturing | Defect detection | Defective / Good |
| Sentiment | Review analysis | Positive / Negative / Neutral |

### Industry Example: Credit Card Fraud Detection

```python
# Features used in fraud detection
features = [
    'transaction_amount',
    'time_since_last_transaction',
    'distance_from_home',
    'merchant_category',
    'device_used_before',
    'ip_country_match'
]

# The challenge: Imbalanced classes
# - 99.9% legitimate transactions
# - 0.1% fraudulent transactions

# Solutions:
# - Oversampling minority class (SMOTE)
# - Undersampling majority class
# - Adjusting class weights
# - Using precision/recall instead of accuracy
```

## Common Pitfalls

1. **Imbalanced Classes**: When one class dominates (99% vs 1%), accuracy is misleading
   - A model predicting "not fraud" always gets 99% accuracy but catches no fraud!
   - Use balanced metrics: precision, recall, F1, AUC-ROC

2. **Incorrect Threshold**: Default 0.5 threshold may not be optimal
   - Lower threshold catches more positives (higher recall)
   - Higher threshold reduces false positives (higher precision)

3. **Ignoring Feature Scaling**: Many algorithms (logistic regression, SVM, neural networks) require scaled features
   - Features with larger ranges dominate the decision

4. **Overfitting to Training Data**: Model memorizes training examples
   - Always evaluate on held-out test data
   - Use cross-validation for robust estimates

## Key Takeaways

1. **Classification predicts discrete categories** - not continuous values.

2. **Decision boundaries separate classes** - the classifier learns where to draw these lines.

3. **Logistic regression outputs probabilities** - the sigmoid function bounds output between 0 and 1.

4. **Accuracy isn't everything** - consider precision, recall, and F1 for imbalanced problems.

5. **Many algorithms exist** - logistic regression is interpretable; neural networks handle complex patterns.

## Looking Ahead

You've now seen both pillars of supervised learning: regression and classification. The next readings introduce **unsupervised learning** through K-Means clustering and distance metrics - algorithms that discover patterns without any labels at all.

By the end of the week, you'll build neural network classifiers that can recognize handwritten digits and fashion items - a dramatic leap from logistic regression's linear decision boundaries.

## Additional Resources

- [Logistic Regression - StatQuest](https://www.youtube.com/watch?v=yIYKR4sgzI8) - Clear visual explanation
- [scikit-learn Classification](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning) - Algorithm overview
- [ROC Curves Explained](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc) - Google's guide to evaluation metrics

