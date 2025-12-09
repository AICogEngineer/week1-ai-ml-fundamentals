# Exercise 12: Training Analysis Diagnostic Lab

## Learning Objectives

- Interpret training and validation curves
- Identify overfitting, underfitting, and other issues
- Recommend remedies for common training problems
- Develop practical debugging skills

## Duration

**Estimated Time:** 60 minutes

## Type

**Diagnostic Lab:** Analyze provided training histories and recommend fixes

---

## The Scenario

You're a ML engineer reviewing training runs from your team. For each run, diagnose the problem (if any) and recommend fixes.

---

## Part 1: Understanding Healthy Training (10 min)

A healthy training run looks like this:

```
Characteristics:
- Loss decreases on both training and validation
- Accuracy increases on both training and validation
- Training and validation metrics stay close together
- Metrics plateau (stop improving) toward the end
```

Example of healthy curves:
```
Epoch 1:  train_loss=0.8, val_loss=0.9
Epoch 5:  train_loss=0.4, val_loss=0.45
Epoch 10: train_loss=0.2, val_loss=0.25
Epoch 15: train_loss=0.15, val_loss=0.20  # Still close!
```

---

## Part 2: Diagnose These Training Runs

### Case 1: The Overfit Model

```python
# Training history from Model A
history_a = {
    'loss':     [0.8, 0.5, 0.3, 0.15, 0.08, 0.03, 0.01, 0.005],
    'val_loss': [0.9, 0.6, 0.5, 0.55, 0.65, 0.80, 0.95, 1.10],
    'accuracy':     [0.70, 0.82, 0.90, 0.95, 0.98, 0.99, 0.998, 0.999],
    'val_accuracy': [0.68, 0.78, 0.82, 0.81, 0.79, 0.77, 0.75, 0.73]
}
```

**Your Analysis:**
```
1. What is happening? (Diagnosis)
   Answer:

2. At which epoch should training have stopped?
   Answer:

3. What remedies would you recommend? (List at least 3)
   - Remedy 1:
   - Remedy 2:
   - Remedy 3:
```

### Case 2: The Underfit Model

```python
# Training history from Model B
history_b = {
    'loss':     [0.9, 0.85, 0.82, 0.80, 0.79, 0.78, 0.77, 0.76],
    'val_loss': [0.95, 0.90, 0.88, 0.86, 0.85, 0.84, 0.83, 0.82],
    'accuracy':     [0.30, 0.35, 0.38, 0.40, 0.41, 0.42, 0.43, 0.44],
    'val_accuracy': [0.28, 0.33, 0.36, 0.38, 0.39, 0.40, 0.41, 0.42]
}
```

**Your Analysis:**
```
1. What is happening? (Diagnosis)
   Answer:

2. Why are training and validation metrics similar but both poor?
   Answer:

3. What remedies would you recommend? (List at least 3)
   - Remedy 1:
   - Remedy 2:
   - Remedy 3:
```

### Case 3: The Unstable Model

```python
# Training history from Model C
history_c = {
    'loss':     [0.8, 0.3, 0.9, 0.2, 1.1, 0.15, 1.5, 0.1],
    'val_loss': [0.9, 0.5, 1.2, 0.4, 1.8, 0.3, 2.5, 0.25],
    'accuracy':     [0.65, 0.85, 0.60, 0.90, 0.55, 0.92, 0.50, 0.95],
    'val_accuracy': [0.60, 0.78, 0.52, 0.82, 0.45, 0.85, 0.40, 0.88]
}
```

**Your Analysis:**
```
1. What is happening? (Diagnosis)
   Answer:

2. What is causing the oscillation?
   Answer:

3. What remedies would you recommend?
   - Remedy 1:
   - Remedy 2:
```

### Case 4: The Stuck Model

```python
# Training history from Model D
history_d = {
    'loss':     [2.30, 2.30, 2.30, 2.30, 2.30, 2.30, 2.30, 2.30],
    'val_loss': [2.30, 2.30, 2.30, 2.30, 2.30, 2.30, 2.30, 2.30],
    'accuracy':     [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
    'val_accuracy': [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10]
}
```

**Your Analysis:**
```
1. What is happening? (Diagnosis)
   Answer:

2. The accuracy is exactly 10% - what does this suggest for a 10-class problem?
   Answer:

3. What could cause the model to be completely stuck?
   - Possible cause 1:
   - Possible cause 2:
   - Possible cause 3:
```

### Case 5: The Almost Perfect Model

```python
# Training history from Model E
history_e = {
    'loss':     [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002],
    'val_loss': [0.55, 0.25, 0.15, 0.10, 0.08, 0.07, 0.065, 0.062],
    'accuracy':     [0.85, 0.93, 0.97, 0.99, 0.995, 0.998, 0.999, 0.9995],
    'val_accuracy': [0.82, 0.90, 0.94, 0.96, 0.97, 0.975, 0.978, 0.979]
}
```

**Your Analysis:**
```
1. Is this model overfitting? How can you tell?
   Answer:

2. What is the gap between training and validation accuracy?
   Answer:

3. Is this acceptable? Should anything be done?
   Answer:
```

---

## Part 3: Visualize the Cases (15 min)

```python
import matplotlib.pyplot as plt

def plot_training_history(history, title):
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

# TODO: Plot each case
# plot_training_history(history_a, 'Case 1: Overfitting')
# plot_training_history(history_b, 'Case 2: Underfitting')
# ...
```

---

## Part 4: Remedies Reference (10 min)

Complete this remedies table:

| Problem | Symptom | Remedies |
|---------|---------|----------|
| Overfitting | Train loss << Val loss | 1. _______ 2. _______ 3. _______ |
| Underfitting | Both losses high | 1. _______ 2. _______ 3. _______ |
| Unstable training | Oscillating metrics | 1. _______ 2. _______ |
| Stuck training | No improvement | 1. _______ 2. _______ 3. _______ |
| Slow convergence | Very gradual improvement | 1. _______ 2. _______ |

---

## Part 5: Real Debugging Scenario (10 min)

You receive this error report:

```
Model: CNN for image classification
Dataset: 50,000 training images, 10,000 test images
Training: 50 epochs
Result: Training accuracy 99.9%, Test accuracy 65%
```

Write a debugging plan:

```
1. First thing to check:

2. Likely diagnosis:

3. Code changes to try (be specific):

4. How to prevent this in the future:
```

---

## Definition of Done

- [ ] All 5 cases diagnosed with explanations
- [ ] Remedies listed for each case
- [ ] Visualizations created
- [ ] Remedies reference table completed
- [ ] Real debugging scenario answered

---

## Key Takeaways

After completing this exercise, you should be able to:

1. **Recognize overfitting:** Diverging train/val curves
2. **Recognize underfitting:** Both metrics poor and similar
3. **Recognize instability:** Oscillating metrics
4. **Apply remedies:** Know which technique fixes which problem
5. **Use early stopping:** Stop before overfitting gets worse

