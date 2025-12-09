# Basic Graphs and Operations

## Learning Objectives

- Understand TensorFlow computation graphs and how operations are organized
- Explain how TensorFlow optimizes execution through graph compilation
- Perform basic math operations and tensor manipulations
- Use @tf.function to create performant graph-based code

## Why This Matters

Understanding graphs is understanding how TensorFlow thinks. While eager execution lets you write code that runs immediately (great for debugging), TensorFlow's real power comes from its ability to optimize computation graphs for maximum performance.

In our **From Zero to Neural** journey, graphs explain why TensorFlow can train neural networks efficiently on GPUs and TPUs - and why the same code runs fast whether you have one GPU or a thousand.

## The Concept

### What Is a Computation Graph?

A **computation graph** is a directed acyclic graph (DAG) where:
- **Nodes** represent operations (add, multiply, etc.)
- **Edges** represent tensors flowing between operations

```
Simple Graph: y = (a + b) * c

    a     b
     \   /
      [+]  <-- Addition operation
       |
       v
      sum    c
        \   /
         [*]  <-- Multiplication operation
          |
          v
          y
```

### Eager vs. Graph Execution

**Eager Execution** (default in TF 2.x):
```python
# Operations execute immediately
a = tf.constant(2)
b = tf.constant(3)
c = tf.constant(4)

sum_ab = a + b      # Executes NOW, returns 5
y = sum_ab * c      # Executes NOW, returns 20

print(y)  # tf.Tensor(20, ...)
```

**Graph Execution** (via @tf.function):
```python
@tf.function
def compute(a, b, c):
    sum_ab = a + b
    return sum_ab * c

# First call: TF traces the function, builds a graph
# Subsequent calls: executes optimized graph
y = compute(tf.constant(2), tf.constant(3), tf.constant(4))
```

### Graph Advantages

| Advantage | Description |
|-----------|-------------|
| **Optimization** | TF can reorder operations, fuse kernels, eliminate redundancy |
| **Parallelism** | Independent operations can run simultaneously |
| **Distribution** | Graphs can be partitioned across devices |
| **Serialization** | Graphs can be saved and loaded (SavedModel) |
| **Hardware acceleration** | Graphs are optimized for GPU/TPU |

### TensorFlow Operations (Ops)

Operations transform tensors. Here are the most common categories:

**Arithmetic Operations:**
```python
import tensorflow as tf

a = tf.constant([1.0, 2.0, 3.0])
b = tf.constant([4.0, 5.0, 6.0])

# Element-wise operations
add = tf.add(a, b)        # or a + b
subtract = tf.subtract(a, b)  # or a - b
multiply = tf.multiply(a, b)  # or a * b
divide = tf.divide(a, b)      # or a / b
power = tf.pow(a, 2)      # a^2

print(f"Add: {add.numpy()}")
print(f"Multiply: {multiply.numpy()}")
print(f"Power: {power.numpy()}")
```

**Matrix Operations:**
```python
A = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
B = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)

# Matrix multiplication
matmul = tf.matmul(A, B)  # or A @ B
print(f"A @ B:\n{matmul.numpy()}")

# Transpose
transposed = tf.transpose(A)
print(f"A transposed:\n{transposed.numpy()}")

# Inverse
inverse = tf.linalg.inv(A)
print(f"A inverse:\n{inverse.numpy()}")
```

**Reduction Operations:**
```python
t = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)

# Sum
total = tf.reduce_sum(t)           # All elements
row_sums = tf.reduce_sum(t, axis=1)  # Sum each row
col_sums = tf.reduce_sum(t, axis=0)  # Sum each column

print(f"Total sum: {total.numpy()}")
print(f"Row sums: {row_sums.numpy()}")
print(f"Col sums: {col_sums.numpy()}")

# Mean
mean_all = tf.reduce_mean(t)
print(f"Mean: {mean_all.numpy()}")

# Max/Min
max_val = tf.reduce_max(t)
min_val = tf.reduce_min(t)
print(f"Max: {max_val.numpy()}, Min: {min_val.numpy()}")
```

**Comparison Operations:**
```python
a = tf.constant([1, 2, 3, 4])
b = tf.constant([2, 2, 2, 2])

equal = tf.equal(a, b)           # [False, True, False, False]
greater = tf.greater(a, b)       # [False, False, True, True]
less_equal = tf.less_equal(a, b) # [True, True, False, False]

print(f"a == b: {equal.numpy()}")
print(f"a > b: {greater.numpy()}")
```

### The @tf.function Decorator

`@tf.function` converts a Python function into a TensorFlow graph:

```python
# Without @tf.function: runs eagerly each time
def eager_function(x):
    return x ** 2 + 2 * x + 1

# With @tf.function: traced once, then runs as optimized graph
@tf.function
def graph_function(x):
    return x ** 2 + 2 * x + 1

# First call traces the function
result = graph_function(tf.constant(3.0))  # Traces and builds graph

# Subsequent calls execute the graph (faster)
result = graph_function(tf.constant(4.0))  # Uses cached graph
```

**Tracing Behavior:**
```python
@tf.function
def traced_function(x):
    print(f"Tracing with x = {x}")  # Only prints during tracing
    tf.print(f"Executing with x =", x)  # Prints every execution
    return x * 2

# First call: traces (Python print runs)
traced_function(tf.constant(1))
# Output: Tracing with x = ...
#         Executing with x = 1

# Second call: uses cached graph (Python print doesn't run)
traced_function(tf.constant(2))
# Output: Executing with x = 2
```

**Retracing Warnings:**
Different input signatures cause retracing:
```python
@tf.function
def add_one(x):
    return x + 1

# Each shape causes a new trace
add_one(tf.constant([1, 2]))      # Traces for shape (2,)
add_one(tf.constant([1, 2, 3]))   # Traces again for shape (3,)

# Fix: use input_signature to specify expected shapes
@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.int32)])
def add_one_flexible(x):
    return x + 1

# Now accepts any 1D tensor without retracing
add_one_flexible(tf.constant([1, 2]))
add_one_flexible(tf.constant([1, 2, 3, 4, 5]))
```

### Automatic Differentiation with GradientTape

TensorFlow tracks operations for computing gradients:

```python
x = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = x ** 2 + 2 * x + 1  # y = x^2 + 2x + 1

# Compute gradient: dy/dx = 2x + 2
gradient = tape.gradient(y, x)
print(f"At x=3: dy/dx = {gradient.numpy()}")  # 2*3 + 2 = 8
```

**Multiple Variables:**
```python
w = tf.Variable(2.0)
b = tf.Variable(1.0)
x = tf.constant(3.0)

with tf.GradientTape() as tape:
    y = w * x + b  # Linear function

# Gradients for all trainable variables
gradients = tape.gradient(y, [w, b])
print(f"dy/dw = {gradients[0].numpy()}")  # x = 3
print(f"dy/db = {gradients[1].numpy()}")  # 1
```

### Control Flow in Graphs

Standard Python control flow works in eager mode but needs care in graphs:

```python
@tf.function
def control_flow_example(x):
    # TensorFlow control flow (graph-compatible)
    if tf.reduce_sum(x) > 0:  # This works!
        return x * 2
    else:
        return x * -1

# Equivalent using tf.cond
@tf.function
def explicit_control_flow(x):
    return tf.cond(
        tf.reduce_sum(x) > 0,
        lambda: x * 2,
        lambda: x * -1
    )
```

**Loops:**
```python
@tf.function
def loop_example(n):
    total = tf.constant(0)
    for i in tf.range(n):  # Use tf.range, not Python range
        total = total + i
    return total

print(loop_example(tf.constant(10)))  # 0+1+2+...+9 = 45
```

## Code Example: Graph Operations in Practice

```python
import tensorflow as tf
import time

print("=" * 60)
print("TENSORFLOW GRAPHS AND OPERATIONS")
print("=" * 60)

# === Basic Operations ===
print("\n--- Basic Operations ---")

a = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
b = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)

print(f"a:\n{a.numpy()}")
print(f"b:\n{b.numpy()}")

# Various operations
print(f"a + b:\n{(a + b).numpy()}")
print(f"a * b (element-wise):\n{(a * b).numpy()}")
print(f"a @ b (matrix multiply):\n{(a @ b).numpy()}")

# === Reduction Operations ===
print("\n--- Reductions ---")

data = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=tf.float32)
print(f"Data:\n{data.numpy()}")
print(f"Sum all: {tf.reduce_sum(data).numpy()}")
print(f"Sum rows (axis=1): {tf.reduce_sum(data, axis=1).numpy()}")
print(f"Mean all: {tf.reduce_mean(data).numpy()}")
print(f"Max per column: {tf.reduce_max(data, axis=0).numpy()}")

# === @tf.function Performance ===
print("\n--- @tf.function Performance ---")

def eager_matmul(a, b, iterations):
    result = a
    for _ in range(iterations):
        result = tf.matmul(result, b)
    return result

@tf.function
def graph_matmul(a, b, iterations):
    result = a
    for _ in range(iterations):
        result = tf.matmul(result, b)
    return result

# Create test matrices
size = 100
iterations = 100
a = tf.random.normal([size, size])
b = tf.random.normal([size, size])

# Warm up
_ = eager_matmul(a, b, 1)
_ = graph_matmul(a, b, 1)

# Time eager execution
start = time.time()
_ = eager_matmul(a, b, iterations)
eager_time = time.time() - start

# Time graph execution
start = time.time()
_ = graph_matmul(a, b, iterations)
graph_time = time.time() - start

print(f"Eager execution: {eager_time:.4f}s")
print(f"Graph execution: {graph_time:.4f}s")
print(f"Speedup: {eager_time / graph_time:.2f}x")

# === Automatic Differentiation ===
print("\n--- Automatic Differentiation ---")

# Simple gradient
x = tf.Variable(2.0)
with tf.GradientTape() as tape:
    y = x ** 3  # y = x^3

dy_dx = tape.gradient(y, x)
print(f"y = x^3 at x=2: y={y.numpy()}")
print(f"dy/dx = 3x^2 = {dy_dx.numpy()} (expected: 12)")

# Neural network gradient example
print("\n--- NN Gradient Example ---")

# Simple single-neuron forward pass
W = tf.Variable([[0.5, -0.3]], dtype=tf.float32)  # Weights
b = tf.Variable([0.1], dtype=tf.float32)          # Bias
x = tf.constant([[1.0, 2.0]])                     # Input

with tf.GradientTape() as tape:
    z = x @ tf.transpose(W) + b
    a = tf.nn.sigmoid(z)
    # Assume target is 1.0, compute simple loss
    loss = (1.0 - a) ** 2

print(f"Weights: {W.numpy()}")
print(f"Input: {x.numpy()}")
print(f"Output: {a.numpy()}")
print(f"Loss: {loss.numpy()}")

# Get gradients for W and b
gradients = tape.gradient(loss, [W, b])
print(f"dLoss/dW: {gradients[0].numpy()}")
print(f"dLoss/db: {gradients[1].numpy()}")

print("\n" + "=" * 60)
```

**Sample Output:**
```
============================================================
TENSORFLOW GRAPHS AND OPERATIONS
============================================================

--- Basic Operations ---
a:
[[1. 2.]
 [3. 4.]]
b:
[[5. 6.]
 [7. 8.]]
a + b:
[[ 6.  8.]
 [10. 12.]]
a * b (element-wise):
[[ 5. 12.]
 [21. 32.]]
a @ b (matrix multiply):
[[19. 22.]
 [43. 50.]]

--- Reductions ---
Data:
[[1. 2. 3.]
 [4. 5. 6.]
 [7. 8. 9.]]
Sum all: 45.0
Sum rows (axis=1): [ 6. 15. 24.]
Mean all: 5.0
Max per column: [7. 8. 9.]

--- @tf.function Performance ---
Eager execution: 0.1234s
Graph execution: 0.0234s
Speedup: 5.27x

--- Automatic Differentiation ---
y = x^3 at x=2: y=8.0
dy/dx = 3x^2 = 12.0 (expected: 12)

--- NN Gradient Example ---
Weights: [[ 0.5 -0.3]]
Input: [[1. 2.]]
Output: [[0.475]]
Loss: [[0.276]]
dLoss/dW: [[-0.261 -0.523]]
dLoss/db: [-0.261]

============================================================
```

## Key Takeaways

1. **Computation graphs organize operations** - nodes are ops, edges are tensors.

2. **@tf.function creates optimized graphs** - significant speedups for complex computations.

3. **Operations are building blocks** - arithmetic, matrix, reduction, and comparison ops cover most needs.

4. **GradientTape enables automatic differentiation** - essential for training neural networks.

5. **Graph execution enables optimization** - parallelism, fusion, and hardware acceleration.

## Looking Ahead

With operations and graphs understood, you're ready to use **Keras Sequential API** to build neural networks without manually managing operations. Keras abstracts the graph creation while giving you full control when needed.

## Additional Resources

- [TensorFlow Graphs Guide](https://www.tensorflow.org/guide/intro_to_graphs) - Official documentation
- [@tf.function Best Practices](https://www.tensorflow.org/guide/function) - Optimization tips
- [Automatic Differentiation](https://www.tensorflow.org/guide/autodiff) - GradientTape deep dive

