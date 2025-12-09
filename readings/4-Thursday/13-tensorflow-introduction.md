# TensorFlow Introduction

## Learning Objectives

- Understand TensorFlow as a machine learning framework and its ecosystem
- Differentiate between eager execution and graph execution modes
- Compare TensorFlow with other frameworks (PyTorch, JAX)
- Set up TensorFlow for development and verify GPU access

## Why This Matters

You've spent the past two days understanding neural networks conceptually and mathematically. Now it's time to build them. TensorFlow is one of the most widely adopted machine learning frameworks in industry, powering everything from Google Search to medical image analysis to self-driving cars.

In our **From Zero to Neural** journey, TensorFlow transforms theory into practice. By the end of today, you'll have the tools to build, train, and deploy neural networks on real data.

## The Concept

### What Is TensorFlow?

**TensorFlow** is an open-source machine learning framework developed by Google Brain. It provides:

1. **Automatic differentiation** - computes gradients for backpropagation automatically
2. **GPU/TPU acceleration** - runs computations on specialized hardware
3. **Production deployment** - tools for serving models in production
4. **Ecosystem** - extensive libraries for various ML tasks

```
TensorFlow's Core Value:

You define:
  - Network architecture (layers)
  - Loss function (what to optimize)
  - Data pipeline

TensorFlow handles:
  - Forward propagation
  - Gradient computation (backprop)
  - Weight updates
  - GPU memory management
  - Distributed training
```

### The TensorFlow Ecosystem

```
TensorFlow Ecosystem

                      TensorFlow Core
                      (Low-level API)
                            |
        +---------------------------------------+
        |                   |                   |
     Keras              TF Extended         TF Lite
   (High-level)        (TFX Pipeline)      (Mobile/IoT)
        |                   |                   |
  - Sequential API    - Data validation    - Model compression
  - Functional API    - Transform          - Edge deployment
  - Model subclass    - Trainer            - On-device ML
                      - Serving
        |
        v
  TensorFlow.js       TensorFlow Hub      TensorBoard
  (Browser ML)        (Pre-trained)       (Visualization)
```

**Key Components:**

| Component | Purpose |
|-----------|---------|
| **TensorFlow Core** | Low-level tensor operations |
| **Keras** | High-level neural network API (built into TF 2.x) |
| **TensorBoard** | Training visualization and monitoring |
| **TF Lite** | Deploy on mobile and edge devices |
| **TFX** | Production ML pipelines |
| **TF Hub** | Pre-trained models repository |

### Eager Execution vs. Graph Execution

TensorFlow 2.x defaults to **eager execution** but supports **graph execution** for performance.

**Eager Execution (Default):**
```python
import tensorflow as tf

# Operations execute immediately
a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])
c = a + b
print(c)  # tf.Tensor([5 7 9], shape=(3,), dtype=int32)

# You can use Python control flow naturally
if c[0] > 3:
    print("First element > 3")
```

Advantages:
- Intuitive debugging
- Python control flow works naturally
- Immediate results

**Graph Execution (for Performance):**
```python
@tf.function  # Decorator converts to graph
def compute(a, b):
    return a + b

# First call: traces and builds graph
# Subsequent calls: executes optimized graph
result = compute(tf.constant([1, 2]), tf.constant([3, 4]))
```

Advantages:
- Optimized execution
- Portable (can save and load graphs)
- Distributed training support

**When to Use Each:**

| Scenario | Mode |
|----------|------|
| Development, debugging | Eager |
| Production training | @tf.function |
| Model deployment | Graph (SavedModel) |
| Quick experiments | Eager |

### TensorFlow vs. Other Frameworks

| Aspect | TensorFlow | PyTorch | JAX |
|--------|------------|---------|-----|
| **Developer** | Google | Meta | Google |
| **Default Mode** | Eager (graph via @tf.function) | Eager | Functional |
| **Deployment** | Excellent (TF Lite, TF.js, TFX) | Growing (TorchScript) | Limited |
| **High-level API** | Keras (built-in) | torch.nn | Flax, Haiku |
| **Debugging** | Good | Excellent | Good |
| **Research Focus** | Production, deployment | Research, flexibility | Research, performance |
| **Industry Adoption** | Very high | High (growing) | Growing |

**When to Choose TensorFlow:**
- Production deployment is critical
- Mobile/edge deployment needed
- Team already uses TensorFlow
- Need extensive ecosystem tools

**When to Choose PyTorch:**
- Research and experimentation
- Dynamic computation graphs
- NLP tasks (historically stronger community)

**JAX:**
- High-performance computing
- Functional programming style
- Research requiring custom differentiation

### Installation and Setup

**Basic Installation:**

```bash
# CPU-only installation
pip install tensorflow

# GPU installation (requires CUDA, cuDNN)
pip install tensorflow[and-cuda]

# Verify installation
python -c "import tensorflow as tf; print(tf.__version__)"
```

**GPU Setup (NVIDIA):**

Requirements:
1. NVIDIA GPU with CUDA support
2. NVIDIA drivers
3. CUDA Toolkit
4. cuDNN library

```python
import tensorflow as tf

# Check TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

# Check if GPU is available
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")

# See compute devices
print("Available devices:")
for device in tf.config.list_physical_devices():
    print(f"  {device}")
```

**Sample Output (with GPU):**
```
TensorFlow version: 2.15.0
GPU available: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
Available devices:
  PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU')
  PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')
```

### Virtual Environments (Best Practice)

```bash
# Create virtual environment
python -m venv tf_env

# Activate (Windows)
tf_env\Scripts\activate

# Activate (Mac/Linux)
source tf_env/bin/activate

# Install TensorFlow
pip install tensorflow

# Create requirements.txt
pip freeze > requirements.txt
```

## Code Example: TensorFlow Basics

```python
import tensorflow as tf
import numpy as np

print("=" * 60)
print("TENSORFLOW BASICS")
print("=" * 60)

# === Version and Device Check ===
print(f"\nTensorFlow Version: {tf.__version__}")
print(f"Eager execution: {tf.executing_eagerly()}")

gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs available: {len(gpus)}")

# === Creating Tensors ===
print("\n--- Creating Tensors ---")

# From Python lists
tensor_from_list = tf.constant([1, 2, 3, 4])
print(f"From list: {tensor_from_list}")

# From NumPy
numpy_array = np.array([[1.0, 2.0], [3.0, 4.0]])
tensor_from_numpy = tf.constant(numpy_array)
print(f"From NumPy:\n{tensor_from_numpy}")

# Special tensors
zeros = tf.zeros([3, 3])
ones = tf.ones([2, 4])
random = tf.random.normal([3, 3], mean=0, stddev=1)

print(f"Zeros shape: {zeros.shape}")
print(f"Ones shape: {ones.shape}")
print(f"Random normal:\n{random}")

# === Tensor Properties ===
print("\n--- Tensor Properties ---")
t = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
print(f"Tensor: {t}")
print(f"Shape: {t.shape}")
print(f"Dtype: {t.dtype}")
print(f"Rank (dimensions): {tf.rank(t)}")
print(f"Total elements: {tf.size(t)}")

# === Basic Operations ===
print("\n--- Basic Operations ---")
a = tf.constant([1.0, 2.0, 3.0])
b = tf.constant([4.0, 5.0, 6.0])

print(f"a = {a.numpy()}")
print(f"b = {b.numpy()}")
print(f"a + b = {(a + b).numpy()}")
print(f"a * b = {(a * b).numpy()}")  # Element-wise
print(f"a @ b (dot) = {tf.tensordot(a, b, 1).numpy()}")

# Matrix multiplication
A = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
B = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)
print(f"\nMatrix A:\n{A.numpy()}")
print(f"Matrix B:\n{B.numpy()}")
print(f"A @ B:\n{(A @ B).numpy()}")

# === Variables (for training) ===
print("\n--- TensorFlow Variables ---")
# Variables are mutable tensors (used for weights)
weights = tf.Variable(tf.random.normal([3, 2]))
print(f"Initial weights:\n{weights.numpy()}")

# Update variables
weights.assign(tf.zeros([3, 2]))
print(f"After assign zeros:\n{weights.numpy()}")

weights.assign_add(tf.ones([3, 2]))
print(f"After assign_add ones:\n{weights.numpy()}")

# === Automatic Differentiation ===
print("\n--- Automatic Differentiation (GradientTape) ---")
x = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = x ** 2  # y = x^2

# dy/dx = 2x = 2*3 = 6
gradient = tape.gradient(y, x)
print(f"x = {x.numpy()}")
print(f"y = x^2 = {y.numpy()}")
print(f"dy/dx = {gradient.numpy()} (expected: 2*x = 6)")

print("\n" + "=" * 60)
```

**Sample Output:**
```
============================================================
TENSORFLOW BASICS
============================================================

TensorFlow Version: 2.15.0
Eager execution: True
GPUs available: 1

--- Creating Tensors ---
From list: [1 2 3 4]
From NumPy:
[[1. 2.]
 [3. 4.]]
Zeros shape: (3, 3)
Ones shape: (2, 4)
Random normal:
[[-0.23  0.87  1.12]
 [ 0.45 -0.67  0.34]
 [ 0.89  0.12 -0.56]]

--- Tensor Properties ---
Tensor: [[1. 2. 3.]
 [4. 5. 6.]]
Shape: (2, 3)
Dtype: <dtype: 'float32'>
Rank (dimensions): 2
Total elements: 6

--- Basic Operations ---
a = [1. 2. 3.]
b = [4. 5. 6.]
a + b = [5. 7. 9.]
a * b = [ 4. 10. 18.]
a @ b (dot) = 32.0

Matrix A:
[[1. 2.]
 [3. 4.]]
Matrix B:
[[5. 6.]
 [7. 8.]]
A @ B:
[[19. 22.]
 [43. 50.]]

--- TensorFlow Variables ---
Initial weights:
[[ 0.23 -0.45]
 [ 0.12  0.78]
 [-0.34  0.56]]
After assign zeros:
[[0. 0.]
 [0. 0.]
 [0. 0.]]
After assign_add ones:
[[1. 1.]
 [1. 1.]
 [1. 1.]]

--- Automatic Differentiation (GradientTape) ---
x = 3.0
y = x^2 = 9.0
dy/dx = 6.0 (expected: 2*x = 6)

============================================================
```

### TensorFlow Programming Model

```
TensorFlow Workflow:

1. Define Data Pipeline
   - Load data
   - Preprocess
   - Batch
   - Shuffle

2. Build Model
   - Define layers
   - Connect layers
   - Specify activations

3. Compile Model
   - Choose optimizer (Adam, SGD)
   - Choose loss function
   - Choose metrics

4. Train Model
   - model.fit(data, labels, epochs)
   - Monitor with callbacks

5. Evaluate & Deploy
   - model.evaluate()
   - model.save()
   - TF Serving / TF Lite
```

## Key Takeaways

1. **TensorFlow is an end-to-end ML platform** - from research to production deployment.

2. **Keras is the high-level API** - built into TensorFlow 2.x for easy model building.

3. **Eager execution is default** - operations execute immediately; use @tf.function for performance.

4. **GPU acceleration is automatic** - TensorFlow uses available GPUs without code changes.

5. **GradientTape enables automatic differentiation** - the foundation for backpropagation.

## Looking Ahead

Now that you have TensorFlow set up and understand its basics, the next readings dive into:
- **Tensors and Shapes** - the data structures that flow through networks
- **Graphs and Operations** - how TensorFlow executes computations
- **Keras Sequential API** - building neural networks easily
- **Dense Layers** - implementing the MLPs you learned about yesterday

## Additional Resources

- [TensorFlow Official Tutorials](https://www.tensorflow.org/tutorials) - Step-by-step guides
- [TensorFlow API Documentation](https://www.tensorflow.org/api_docs/python/tf) - Comprehensive reference
- [TensorFlow vs PyTorch](https://www.tensorflow.org/guide/migrate) - Migration and comparison guide

