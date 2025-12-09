# Visualizing Training Metrics

## Learning Objectives

- Interpret training loss and accuracy curves
- Detect overfitting and underfitting from plots
- Use TensorBoard for interactive training monitoring
- Apply matplotlib for custom training visualizations

## Why This Matters

Training a neural network without monitoring is like driving with your eyes closed. Loss curves and accuracy plots tell you whether your network is learning, overfitting, or struggling to converge.

In our **From Zero to Neural** journey, this final reading equips you to diagnose training problems and make informed decisions about when to stop training, adjust hyperparameters, or modify architecture.

Congratulations - after this reading, you'll have completed Week 1's foundational journey through AI/ML fundamentals!

## The Concept

### What to Monitor During Training

Every training run produces metrics you should track:

| Metric | What It Shows |
|--------|--------------|
| **Training Loss** | How well model fits training data |
| **Validation Loss** | How well model generalizes |
| **Training Accuracy** | Correct predictions on training data |
| **Validation Accuracy** | Correct predictions on unseen data |

### The Ideal Learning Curve

```
Loss                              Accuracy
  |                                 |
  |\                                |         _______
  | \                               |        /
  |  \                              |       /
  |   \_____ Training               |      /
  |    \____ Validation             |     /
  |                                 |    /
  +-----------> Epochs              +-----------> Epochs

Both losses decrease together;    Accuracy increases and plateaus
Validation stays close to         for both training and validation.
training.
```

### Detecting Overfitting

**Overfitting**: Model memorizes training data but fails on new data.

```
Loss (Overfitting Pattern)         Accuracy
  |                                 |
  |\                                |        _______ Training
  | \   Validation                  |       /
  |  \   /----------                |      /   _____ Validation
  |   \_/                           |     /   /
  |    \_____ Training              |    /   /
  |                                 |   /  _/
  +-----------> Epochs              +-----------> Epochs

Validation loss increases while    Training accuracy keeps improving
training loss decreases.           but validation plateaus or drops.
Gap between curves = overfitting!
```

**Signs of Overfitting:**
- Validation loss starts increasing while training loss decreases
- Training accuracy >> Validation accuracy
- Large gap between training and validation metrics

**Solutions:**
- More training data
- Data augmentation
- Regularization (L1/L2, Dropout)
- Reduce model complexity
- Early stopping

### Detecting Underfitting

**Underfitting**: Model too simple to learn the patterns.

```
Loss (Underfitting Pattern)        Accuracy
  |                                 |
  |\                                |
  | \                               |
  |  \_______ Training              |      __________
  |   \______ Validation            |     /
  |                                 |    /  (plateaus low)
  |    (both plateau high)          |   /
  +-----------> Epochs              +-----------> Epochs

Both losses remain high.           Accuracy plateaus at low value.
```

**Signs of Underfitting:**
- Both training and validation loss remain high
- Both accuracies plateau at low values
- Model isn't learning the basic patterns

**Solutions:**
- Increase model complexity (more layers/neurons)
- Train longer
- Reduce regularization
- Try different architecture

### Good Fit

```
Loss (Good Fit)                    Accuracy
  |                                 |
  |\                                |
  | \                               |         _______ Training
  |  \__                            |        /______ Validation
  |    \_____ Training              |       /
  |     \____ Validation            |      /
  |          (close together)       |     /  (close together)
  +-----------> Epochs              +-----------> Epochs

Both losses decrease and           Both accuracies increase and
converge to similar low values.    converge to similar high values.
```

### Using TensorBoard

TensorBoard is TensorFlow's visualization toolkit:

```python
import tensorflow as tf
from tensorflow import keras

# Create TensorBoard callback
tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir='./logs',
    histogram_freq=1,  # Log weight histograms
    write_graph=True,  # Log model graph
    update_freq='epoch'
)

# Add to model.fit()
model.fit(
    X_train, y_train,
    epochs=20,
    validation_data=(X_val, y_val),
    callbacks=[tensorboard_callback]
)
```

**Launch TensorBoard:**
```bash
tensorboard --logdir=./logs
# Open browser to http://localhost:6006
```

**TensorBoard Features:**
- Interactive loss/accuracy plots
- Model graph visualization
- Weight histograms
- Embedding projector
- Hyperparameter comparison

### Using Matplotlib

For custom visualizations in notebooks:

```python
import matplotlib.pyplot as plt

def plot_training_history(history):
    """Plot training and validation metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    axes[1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[1].set_title('Model Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()
```

### Early Stopping

Automatically stop training when validation loss stops improving:

```python
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',      # Metric to watch
    patience=5,              # Epochs to wait for improvement
    restore_best_weights=True  # Restore weights from best epoch
)

model.fit(
    X_train, y_train,
    epochs=100,  # Set high; early stopping will stop when appropriate
    validation_data=(X_val, y_val),
    callbacks=[early_stopping]
)
```

## Code Example: Complete Training Visualization

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("TRAINING VISUALIZATION")
print("=" * 60)

# === Create Dataset ===
print("\n--- Creating Dataset ---")

# MNIST-like synthetic data
np.random.seed(42)
n_samples = 5000
n_features = 784
n_classes = 10

X = np.random.randn(n_samples, n_features).astype(np.float32)
y = np.random.randint(0, n_classes, n_samples)

# Split
X_train, X_val = X[:4000], X[4000:]
y_train, y_val = y[:4000], y[4000:]

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")

# === Build Model ===
print("\n--- Building Model ---")

model = keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=(n_features,)),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# === Train Model ===
print("\n--- Training Model ---")

history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1
)

# === Plot Training History ===
print("\n--- Training History ---")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss
epochs = range(1, len(history.history['loss']) + 1)
axes[0].plot(epochs, history.history['loss'], 'b-', label='Training Loss', linewidth=2)
axes[0].plot(epochs, history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
axes[0].set_title('Training and Validation Loss', fontsize=14)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy
axes[1].plot(epochs, history.history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
axes[1].plot(epochs, history.history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
axes[1].set_title('Training and Validation Accuracy', fontsize=14)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=100)
print("Saved: training_history.png")

# === Analyze Training ===
print("\n--- Training Analysis ---")

final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]

best_val_loss_epoch = np.argmin(history.history['val_loss']) + 1
best_val_loss = min(history.history['val_loss'])

print(f"Final Training Loss: {final_train_loss:.4f}")
print(f"Final Validation Loss: {final_val_loss:.4f}")
print(f"Final Training Accuracy: {final_train_acc:.2%}")
print(f"Final Validation Accuracy: {final_val_acc:.2%}")
print(f"Best Validation Loss: {best_val_loss:.4f} at epoch {best_val_loss_epoch}")

# Check for overfitting
loss_gap = final_val_loss - final_train_loss
acc_gap = final_train_acc - final_val_acc

print(f"\nLoss Gap (val - train): {loss_gap:.4f}")
print(f"Accuracy Gap (train - val): {acc_gap:.2%}")

if loss_gap > 0.5:
    print("Warning: Possible overfitting detected!")
elif final_val_loss > 2.0:
    print("Warning: Model may be underfitting!")
else:
    print("Training appears healthy.")

# === Demonstrate Early Stopping ===
print("\n--- Early Stopping Example ---")

model2 = keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=(n_features,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(n_classes, activation='softmax')
])

model2.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

history2 = model2.fit(
    X_train, y_train,
    epochs=100,  # High epochs - early stopping will cut off
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    verbose=0
)

print(f"Training stopped at epoch: {len(history2.history['loss'])}")
print(f"Best weights restored from epoch: {np.argmin(history2.history['val_loss']) + 1}")

# === Learning Rate Effects ===
print("\n--- Learning Rate Comparison ---")

learning_rates = [0.001, 0.01, 0.1]
histories = {}

for lr in learning_rates:
    model_lr = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(n_features,)),
        layers.Dense(n_classes, activation='softmax')
    ])
    
    model_lr.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    hist = model_lr.fit(
        X_train[:1000], y_train[:1000],  # Smaller subset for speed
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    
    histories[lr] = hist.history
    print(f"LR={lr}: Final val_loss={hist.history['val_loss'][-1]:.4f}")

# Plot LR comparison
fig, ax = plt.subplots(figsize=(10, 6))
for lr, hist in histories.items():
    ax.plot(hist['val_loss'], label=f'LR={lr}')

ax.set_title('Learning Rate Comparison')
ax.set_xlabel('Epoch')
ax.set_ylabel('Validation Loss')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('lr_comparison.png', dpi=100)
print("\nSaved: lr_comparison.png")

print("\n" + "=" * 60)
print("WEEK 1 COMPLETE!")
print("=" * 60)
```

### Interpreting Real Training Curves

**Scenario 1: Smooth Convergence (Good)**
```
Loss drops smoothly, validation follows training closely.
Action: Training is working well. Continue or stop at plateau.
```

**Scenario 2: Oscillating Loss (Too High LR)**
```
Loss jumps up and down erratically.
Action: Reduce learning rate.
```

**Scenario 3: Flat Loss (Too Low LR or Bad Architecture)**
```
Loss barely changes from initial value.
Action: Increase learning rate or check model architecture.
```

**Scenario 4: Diverging Curves (Overfitting)**
```
Training loss decreases but validation loss increases.
Action: Add regularization, get more data, or simplify model.
```

### Checkpointing Best Models

Save the best model during training:

```python
checkpoint = keras.callbacks.ModelCheckpoint(
    filepath='best_model.keras',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

model.fit(
    X_train, y_train,
    epochs=50,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint, early_stopping]
)

# Load best model
best_model = keras.models.load_model('best_model.keras')
```

## Key Takeaways

1. **Monitor both training and validation metrics** - the gap between them tells the story.

2. **Overfitting = diverging curves** - validation gets worse while training improves.

3. **Underfitting = high loss plateau** - both metrics stuck at poor values.

4. **Use early stopping** - automatically stop when validation stops improving.

5. **TensorBoard and matplotlib** - essential tools for understanding your model.

## Week 1 Complete!

Congratulations! You've completed the **From Zero to Neural** journey:

- **Tuesday**: ML algorithms, supervised/unsupervised learning, regression, classification, clustering, distance metrics
- **Wednesday**: Neural networks, perceptrons, activation functions, MLPs, forward propagation, loss functions
- **Thursday**: TensorFlow, tensors, graphs, Keras Sequential API, Dense layers
- **Friday**: CNNs, convolutional layers, pooling, flattening, training visualization

You now have the foundational knowledge to build, train, and evaluate neural networks. Week 2 will deepen these foundations with backpropagation, gradient descent, and advanced training techniques.

## Additional Resources

- [TensorBoard Guide](https://www.tensorflow.org/tensorboard/get_started) - Official tutorial
- [Keras Callbacks](https://keras.io/api/callbacks/) - EarlyStopping, ModelCheckpoint, and more
- [Diagnosing ML Models](https://www.coursera.org/learn/machine-learning/lecture/Kont7/evaluating-a-hypothesis) - Andrew Ng's classic lecture

