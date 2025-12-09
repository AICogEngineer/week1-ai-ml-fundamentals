"""
Exercise 02: Classifier Comparison Lab
======================================

Compare multiple classifiers on the Wine dataset.
Complete the TODO sections.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Import classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# =============================================================================
# PART 1: DATA PREPARATION
# =============================================================================

print("=" * 60)
print("CLASSIFIER COMPARISON LAB")
print("=" * 60)

# Load wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Convert to binary: class 0 vs classes 1+2
y_binary = (y > 0).astype(int)

# TODO: Print dataset info
print("\n--- DATASET INFO ---")
# print(f"Number of samples: {???}")
# print(f"Number of features: {???}")
# print(f"Feature names: {wine.feature_names}")
# print(f"Class distribution: {np.bincount(y_binary)}")

# TODO: Train/test split (80/20, stratified)
# X_train, X_test, y_train, y_test = train_test_split(
#     ???, ???, test_size=0.2, random_state=42, stratify=y_binary
# )

# TODO: Scale features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# =============================================================================
# PART 2: TRAIN CLASSIFIERS
# =============================================================================

print("\n--- TRAINING CLASSIFIERS ---")

# Define classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
    'KNN (k=3)': KNeighborsClassifier(n_neighbors=3),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', random_state=42)
}

results = {}

for name, clf in classifiers.items():
    # TODO: Train the classifier
    # clf.fit(X_train_scaled, y_train)
    
    # TODO: Make predictions
    # y_pred = clf.predict(X_test_scaled)
    
    # TODO: Calculate metrics
    # results[name] = {
    #     'accuracy': accuracy_score(y_test, y_pred),
    #     'precision': precision_score(y_test, y_pred),
    #     'recall': recall_score(y_test, y_pred),
    #     'f1': f1_score(y_test, y_pred)
    # }
    
    # print(f"{name}: Accuracy = {results[name]['accuracy']:.2%}")
    pass

# =============================================================================
# PART 3: ANALYSIS
# =============================================================================

print("\n--- COMPARISON TABLE ---")

# TODO: Create comparison DataFrame
# comparison_df = pd.DataFrame(results).T
# comparison_df = comparison_df.round(4)
# print(comparison_df)

# TODO: Find best classifier
# best_clf = comparison_df['f1'].idxmax()
# print(f"\nBest Classifier: {best_clf} (F1 = {comparison_df.loc[best_clf, 'f1']:.4f})")

# =============================================================================
# PART 4: VISUALIZATION
# =============================================================================

print("\n--- VISUALIZATION ---")

# TODO: Bar chart comparing accuracies
# plt.figure(figsize=(12, 6))
# names = list(results.keys())
# accuracies = [results[n]['accuracy'] for n in names]
# plt.bar(names, accuracies)
# plt.ylabel('Accuracy')
# plt.title('Classifier Comparison')
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.show()

# =============================================================================
# ANALYSIS QUESTIONS
# =============================================================================

# Q1: Which classifier performed best? Why?
# Answer: 

# Q2: Did scaling affect KNN more than Decision Tree? Why?
# Answer: 

# Q3: For costly false positives, which classifier? For costly false negatives?
# Answer: 

# Q4: Trade-off between Decision Tree and Random Forest?
# Answer: 

