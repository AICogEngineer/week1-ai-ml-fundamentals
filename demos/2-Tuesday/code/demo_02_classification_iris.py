"""
Demo 02: Classification with the Iris Dataset
==============================================
Week 1, Tuesday - AI/ML Fundamentals

This demo introduces classification - predicting CATEGORIES instead of numbers.
We use the famous Iris dataset to classify flower species.

INSTRUCTOR NOTES:
- Contrast with regression: "Now we predict WHICH category, not HOW MUCH"
- The Iris dataset is a ML classic - trainees will see it everywhere
- Emphasize the decision boundary concept

Estimated Time: 15-20 minutes
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =============================================================================
# SECTION 1: LOAD AND EXPLORE THE DATA
# =============================================================================
print("=" * 60)
print("DEMO 02: CLASSIFICATION WITH IRIS DATASET")
print("Predicting Flower Species from Measurements")
print("=" * 60)

# Load the famous Iris dataset
iris = load_iris()

# Features and targets
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print("\n--- THE IRIS DATASET ---")
print(f"Number of samples: {len(X)}")
print(f"Number of features: {X.shape[1]}")
print(f"Number of classes: {len(target_names)}")

print(f"\nFeatures: {feature_names}")
print(f"Classes: {list(target_names)}")

print("\nFirst 5 samples:")
print(f"{'Sepal L':<10} {'Sepal W':<10} {'Petal L':<10} {'Petal W':<10} {'Species':<15}")
print("-" * 55)
for i in range(5):
    print(f"{X[i,0]:<10.1f} {X[i,1]:<10.1f} {X[i,2]:<10.1f} {X[i,3]:<10.1f} {target_names[y[i]]:<15}")

# =============================================================================
# SECTION 2: VISUALIZE THE DATA
# =============================================================================
print("\n--- VISUALIZING THE DATA ---")

# Use petal length and width for 2D visualization
X_2d = X[:, 2:4]  # Petal length, petal width

plt.figure(figsize=(10, 6))
colors = ['red', 'green', 'blue']
markers = ['o', 's', '^']

for i, (color, marker, name) in enumerate(zip(colors, markers, target_names)):
    mask = y == i
    plt.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                c=color, marker=marker, label=name, 
                alpha=0.7, edgecolors='black', linewidth=0.5, s=60)

plt.xlabel('Petal Length (cm)', fontsize=12)
plt.ylabel('Petal Width (cm)', fontsize=12)
plt.title('Iris Dataset: Can We Separate the Species?', fontsize=14)
plt.legend(title='Species')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('02_iris_scatter.png', dpi=100)
plt.show()

print("[Saved: 02_iris_scatter.png]")
print("DISCUSSION: Can you draw lines to separate the three species?")

# =============================================================================
# SECTION 3: PREPARE DATA
# =============================================================================
print("\n--- PREPARING DATA ---")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)} (70%)")
print(f"Testing samples: {len(X_test)} (30%)")
print(f"\nNote: stratify=y ensures balanced classes in both sets")

# Class distribution
print("\nClass distribution in training set:")
for i, name in enumerate(target_names):
    count = (y_train == i).sum()
    print(f"  {name}: {count} samples")

# =============================================================================
# SECTION 4: TRAIN CLASSIFIERS
# =============================================================================
print("\n--- TRAINING CLASSIFIERS ---")

# We'll compare three different classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=200),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(max_depth=4, random_state=42)
}

results = {}

for name, clf in classifiers.items():
    print(f"\nTraining {name}...")
    clf.fit(X_train, y_train)
    
    # Predictions
    y_pred = clf.predict(X_test)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = {'model': clf, 'predictions': y_pred, 'accuracy': accuracy}
    
    print(f"  Accuracy: {accuracy:.2%}")

# =============================================================================
# SECTION 5: DEEP DIVE INTO LOGISTIC REGRESSION
# =============================================================================
print("\n--- LOGISTIC REGRESSION DETAILS ---")

log_reg = results['Logistic Regression']['model']
y_pred_lr = results['Logistic Regression']['predictions']

# Show probabilities (not just predictions)
print("Logistic Regression outputs PROBABILITIES:")
print("\nFirst 5 test samples:")
probs = log_reg.predict_proba(X_test[:5])

print(f"{'Setosa':<10} {'Versicolor':<12} {'Virginica':<12} {'Predicted':<12} {'Actual'}")
print("-" * 60)
for i in range(5):
    pred_name = target_names[y_pred_lr[i]]
    actual_name = target_names[y_test[i]]
    print(f"{probs[i,0]:<10.3f} {probs[i,1]:<12.3f} {probs[i,2]:<12.3f} {pred_name:<12} {actual_name}")

# =============================================================================
# SECTION 6: CONFUSION MATRIX
# =============================================================================
print("\n--- CONFUSION MATRIX ---")

cm = confusion_matrix(y_test, y_pred_lr)

print("Confusion Matrix (Logistic Regression):")
print(f"\n{'Predicted:':<12}", end="")
for name in target_names:
    print(f"{name[:8]:<10}", end="")
print()
print("-" * 42)

for i, row in enumerate(cm):
    print(f"{target_names[i][:10]:<12}", end="")
    for val in row:
        print(f"{val:<10}", end="")
    print()

print("""
Reading the confusion matrix:
- Diagonal = correct predictions
- Off-diagonal = errors (which classes get confused?)
""")

# =============================================================================
# SECTION 7: VISUALIZE DECISION BOUNDARY
# =============================================================================
print("\n--- DECISION BOUNDARY ---")

# Use only 2 features for visualization
X_2d_train = X_train[:, 2:4]
X_2d_test = X_test[:, 2:4]

# Train on 2D data
clf_2d = LogisticRegression(max_iter=200)
clf_2d.fit(X_2d_train, y_train)

# Create mesh grid for decision boundary
x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

# Predict on mesh
Z = clf_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
plt.contour(xx, yy, Z, colors='black', linewidths=0.5)

# Plot points
for i, (color, marker, name) in enumerate(zip(colors, markers, target_names)):
    mask = y == i
    plt.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                c=color, marker=marker, label=name, 
                alpha=0.8, edgecolors='black', linewidth=0.5, s=60)

plt.xlabel('Petal Length (cm)', fontsize=12)
plt.ylabel('Petal Width (cm)', fontsize=12)
plt.title('Decision Boundaries: How the Classifier Separates Classes', fontsize=14)
plt.legend(title='Species', loc='upper left')
plt.tight_layout()
plt.savefig('02_decision_boundary.png', dpi=100)
plt.show()

print("[Saved: 02_decision_boundary.png]")

# =============================================================================
# SECTION 8: COMPARE CLASSIFIERS
# =============================================================================
print("\n--- CLASSIFIER COMPARISON ---")

print(f"\n{'Classifier':<25} {'Accuracy':<15}")
print("-" * 40)
for name, data in results.items():
    print(f"{name:<25} {data['accuracy']:.2%}")

# Visualize comparison
plt.figure(figsize=(8, 5))
names = list(results.keys())
accuracies = [results[n]['accuracy'] for n in names]
colors_bar = ['#4CAF50', '#2196F3', '#FF9800']

bars = plt.bar(names, accuracies, color=colors_bar, edgecolor='black')
plt.ylabel('Accuracy', fontsize=12)
plt.title('Classifier Comparison on Iris Dataset', fontsize=14)
plt.ylim(0.8, 1.0)

# Add value labels
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{acc:.1%}', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('02_classifier_comparison.png', dpi=100)
plt.show()

print("[Saved: 02_classifier_comparison.png]")

# =============================================================================
# SECTION 9: CLASSIFICATION REPORT
# =============================================================================
print("\n--- DETAILED CLASSIFICATION REPORT ---")
print("\nLogistic Regression Performance:")
print(classification_report(y_test, y_pred_lr, target_names=target_names))

# =============================================================================
# SECTION 10: KEY TAKEAWAYS
# =============================================================================
print("\n" + "=" * 60)
print("KEY TAKEAWAYS")
print("=" * 60)
print("""
1. CLASSIFICATION vs REGRESSION:
   - Regression predicts continuous values (price, temperature)
   - Classification predicts discrete categories (species, spam/not spam)

2. DECISION BOUNDARIES:
   - Classifiers learn boundaries that separate classes
   - Different algorithms create different boundary shapes

3. PROBABILITIES:
   - Logistic regression outputs probability for each class
   - Prediction = class with highest probability

4. EVALUATION METRICS:
   - Accuracy: % of correct predictions
   - Confusion Matrix: shows which classes get confused
   - Precision/Recall: important for imbalanced datasets

5. MULTIPLE CLASSIFIERS:
   - Different algorithms have different strengths
   - Always compare multiple approaches!
""")

print("Next: We'll explore UNSUPERVISED learning with K-Means clustering")

