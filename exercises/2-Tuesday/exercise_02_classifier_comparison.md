# Exercise 02: Classifier Comparison Lab

## Learning Objectives

- Apply multiple classification algorithms to the same dataset
- Evaluate and compare classifier performance
- Understand when different algorithms excel
- Practice the ML workflow: split, train, predict, evaluate

## Duration

**Estimated Time:** 60 minutes

## Background

In `demo_02_classification_iris.py`, we used Logistic Regression, KNN, and Decision Trees. But how do you choose which one to use in practice? This lab gives you hands-on experience comparing classifiers systematically.

---

## The Dataset: Wine Quality

You'll work with the Wine Quality dataset - predicting whether a wine is "good" (quality >= 7) or "not good" based on chemical properties.

```python
Features: acidity, sugar, pH, alcohol content, etc.
Target: Binary (good/not good)
```

---

## Part 1: Data Preparation (15 min)

### Task 1.1: Load and Explore

Navigate to `starter_code/classifier_comparison.py`:

```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Convert to binary: class 0 vs classes 1+2
y_binary = (y > 0).astype(int)

# TODO: Print dataset info
# - Number of samples
# - Number of features  
# - Class distribution
```

### Task 1.2: Train/Test Split

```python
# TODO: Split data (80% train, 20% test, stratified)
X_train, X_test, y_train, y_test = None  # Your code

# TODO: Scale features (important for KNN and Logistic Regression!)
scaler = StandardScaler()
# Your code here
```

---

## Part 2: Train Multiple Classifiers (20 min)

### Task 2.1: Define Classifiers

```python
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Define classifiers to compare
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
    'KNN (k=3)': KNeighborsClassifier(n_neighbors=3),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', random_state=42)
}
```

### Task 2.2: Train and Evaluate Each

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

results = {}

for name, clf in classifiers.items():
    # TODO: Train the classifier
    # clf.fit(...)
    
    # TODO: Make predictions
    # y_pred = ...
    
    # TODO: Calculate metrics
    results[name] = {
        'accuracy': None,   # accuracy_score(...)
        'precision': None,  # precision_score(...)
        'recall': None,     # recall_score(...)
        'f1': None          # f1_score(...)
    }
    
    print(f"{name}: Accuracy = {results[name]['accuracy']:.2%}")
```

---

## Part 3: Detailed Analysis (15 min)

### Task 3.1: Create Comparison Table

```python
# TODO: Create a DataFrame comparing all classifiers
comparison_df = pd.DataFrame(results).T
comparison_df = comparison_df.round(4)
print(comparison_df)
```

### Task 3.2: Visualize Comparison

```python
import matplotlib.pyplot as plt

# TODO: Create a bar chart comparing accuracies
# plt.figure(figsize=(12, 6))
# ...
```

### Task 3.3: Confusion Matrices

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# TODO: Plot confusion matrix for the BEST and WORST classifier
# Use ConfusionMatrixDisplay.from_estimator(...)
```

---

## Part 4: Analysis Questions (10 min)

Answer these questions in your code comments:

### Q1: Which classifier performed best? Why might that be?

```python
# Your analysis:
```

### Q2: Did scaling affect KNN more than Decision Tree? Why?

```python
# Hint: Try running without scaling and compare
# Your analysis:
```

### Q3: Look at precision vs recall. Which classifier would you choose if:
- False positives are very costly (predicting good wine when it's bad)
- False negatives are very costly (missing good wine)

```python
# Your analysis:
```

### Q4: What's the trade-off between Decision Tree and Random Forest?

```python
# Your analysis:
```

---

## Bonus Challenges

### Challenge A: Cross-Validation

Instead of a single train/test split, use 5-fold cross-validation:

```python
from sklearn.model_selection import cross_val_score

# TODO: Compare classifiers using cross-validation
# cv_scores = cross_val_score(clf, X_scaled, y_binary, cv=5)
```

### Challenge B: Hyperparameter Tuning

Try different values of k for KNN. What's the optimal k?

```python
# TODO: Test k from 1 to 20 and plot accuracy vs k
```

---

## Definition of Done

- [ ] Data loaded and preprocessed correctly
- [ ] At least 5 classifiers trained and evaluated
- [ ] Comparison table created
- [ ] Visualization shows clear comparison
- [ ] All analysis questions answered
- [ ] Best classifier identified with justification

---

## Expected Output Example

```
Classifier Comparison Results:
==================================================
                      accuracy  precision  recall    f1
Logistic Regression     0.9722    0.9600   1.0000  0.9796
KNN (k=5)               0.9444    0.9583   0.9583  0.9583
KNN (k=3)               0.9444    0.9600   0.9600  0.9600
Decision Tree           0.9167    0.9200   0.9583  0.9388
Random Forest           0.9722    0.9615   1.0000  0.9804
SVM (RBF)               0.9722    0.9600   1.0000  0.9796

Best Classifier: Random Forest (F1 = 0.9804)
```

