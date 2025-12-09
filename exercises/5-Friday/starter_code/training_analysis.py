"""
Exercise 12: Training Analysis Diagnostic Lab
=============================================

Analyze training histories and diagnose problems.
"""

import matplotlib.pyplot as plt
import numpy as np

print("=" * 60)
print("TRAINING ANALYSIS DIAGNOSTIC LAB")
print("=" * 60)

# =============================================================================
# CASE 1: THE OVERFIT MODEL
# =============================================================================

history_a = {
    'loss':     [0.8, 0.5, 0.3, 0.15, 0.08, 0.03, 0.01, 0.005],
    'val_loss': [0.9, 0.6, 0.5, 0.55, 0.65, 0.80, 0.95, 1.10],
    'accuracy':     [0.70, 0.82, 0.90, 0.95, 0.98, 0.99, 0.998, 0.999],
    'val_accuracy': [0.68, 0.78, 0.82, 0.81, 0.79, 0.77, 0.75, 0.73]
}

# =============================================================================
# CASE 2: THE UNDERFIT MODEL
# =============================================================================

history_b = {
    'loss':     [0.9, 0.85, 0.82, 0.80, 0.79, 0.78, 0.77, 0.76],
    'val_loss': [0.95, 0.90, 0.88, 0.86, 0.85, 0.84, 0.83, 0.82],
    'accuracy':     [0.30, 0.35, 0.38, 0.40, 0.41, 0.42, 0.43, 0.44],
    'val_accuracy': [0.28, 0.33, 0.36, 0.38, 0.39, 0.40, 0.41, 0.42]
}

# =============================================================================
# CASE 3: THE UNSTABLE MODEL
# =============================================================================

history_c = {
    'loss':     [0.8, 0.3, 0.9, 0.2, 1.1, 0.15, 1.5, 0.1],
    'val_loss': [0.9, 0.5, 1.2, 0.4, 1.8, 0.3, 2.5, 0.25],
    'accuracy':     [0.65, 0.85, 0.60, 0.90, 0.55, 0.92, 0.50, 0.95],
    'val_accuracy': [0.60, 0.78, 0.52, 0.82, 0.45, 0.85, 0.40, 0.88]
}

# =============================================================================
# CASE 4: THE STUCK MODEL
# =============================================================================

history_d = {
    'loss':     [2.30, 2.30, 2.30, 2.30, 2.30, 2.30, 2.30, 2.30],
    'val_loss': [2.30, 2.30, 2.30, 2.30, 2.30, 2.30, 2.30, 2.30],
    'accuracy':     [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
    'val_accuracy': [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10]
}

# =============================================================================
# CASE 5: THE ALMOST PERFECT MODEL
# =============================================================================

history_e = {
    'loss':     [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002],
    'val_loss': [0.55, 0.25, 0.15, 0.10, 0.08, 0.07, 0.065, 0.062],
    'accuracy':     [0.85, 0.93, 0.97, 0.99, 0.995, 0.998, 0.999, 0.9995],
    'val_accuracy': [0.82, 0.90, 0.94, 0.96, 0.97, 0.975, 0.978, 0.979]
}


# =============================================================================
# VISUALIZATION FUNCTION
# =============================================================================

def plot_training_history(history, title):
    """Plot training and validation metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history['loss'], 'b-', label='Training', linewidth=2)
    axes[0].plot(history['val_loss'], 'r-', label='Validation', linewidth=2)
    axes[0].set_title(f'{title} - Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history['accuracy'], 'b-', label='Training', linewidth=2)
    axes[1].plot(history['val_accuracy'], 'r-', label='Validation', linewidth=2)
    axes[1].set_title(f'{title} - Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# =============================================================================
# VISUALIZE ALL CASES
# =============================================================================

print("\n--- VISUALIZING CASES ---")

# TODO: Uncomment to visualize each case
# plot_training_history(history_a, 'Case 1: Overfitting')
# plot_training_history(history_b, 'Case 2: Underfitting')
# plot_training_history(history_c, 'Case 3: Unstable')
# plot_training_history(history_d, 'Case 4: Stuck')
# plot_training_history(history_e, 'Case 5: Good Training')


# =============================================================================
# YOUR ANALYSIS
# =============================================================================

# CASE 1 ANALYSIS:
# Diagnosis:
# Best epoch to stop:
# Remedies:
#   1.
#   2.
#   3.

# CASE 2 ANALYSIS:
# Diagnosis:
# Why both metrics are similar but poor:
# Remedies:
#   1.
#   2.
#   3.

# CASE 3 ANALYSIS:
# Diagnosis:
# Cause of oscillation:
# Remedies:
#   1.
#   2.

# CASE 4 ANALYSIS:
# Diagnosis:
# Why 10% accuracy (for 10 classes):
# Possible causes:
#   1.
#   2.
#   3.

# CASE 5 ANALYSIS:
# Is it overfitting:
# Train/val gap:
# Acceptable:


# =============================================================================
# REMEDIES REFERENCE TABLE
# =============================================================================

# | Problem | Symptom | Remedies |
# |---------|---------|----------|
# | Overfitting | Train loss << Val loss | 1.___ 2.___ 3.___ |
# | Underfitting | Both losses high | 1.___ 2.___ 3.___ |
# | Unstable | Oscillating metrics | 1.___ 2.___ |
# | Stuck | No improvement | 1.___ 2.___ 3.___ |


# =============================================================================
# REAL DEBUGGING SCENARIO
# =============================================================================

# Model: CNN for image classification
# Dataset: 50,000 training, 10,000 test
# Result: Train 99.9%, Test 65%

# 1. First thing to check:
# 2. Likely diagnosis:
# 3. Code changes to try:
# 4. Prevention for future:

