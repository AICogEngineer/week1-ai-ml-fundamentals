# Weekly Knowledge Check: TensorFlow & Keras (Thursday)

## Part 1: Multiple Choice

### 1. What is TensorFlow primarily used for?

- [ ] A) Web development
- [ ] B) Machine learning and deep learning
- [ ] C) Database management
- [ ] D) Operating system development

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Machine learning and deep learning

**Explanation:** TensorFlow is an open-source machine learning framework developed by Google Brain. It provides automatic differentiation, GPU/TPU acceleration, and tools for building, training, and deploying neural networks.

- **Why others are wrong:**
  - A, C, D) TensorFlow is specialized for ML, not general-purpose programming tasks
</details>

---

### 2. What is Keras in relation to TensorFlow 2.x?

- [ ] A) A separate competing framework
- [ ] B) TensorFlow's high-level neural network API built into TF 2.x
- [ ] C) A data visualization tool
- [ ] D) A database for storing models

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) TensorFlow's high-level neural network API built into TF 2.x

**Explanation:** Keras was originally a standalone library but is now the official high-level API integrated into TensorFlow 2.x. It provides intuitive abstractions for building models without writing low-level code.

- **Why others are wrong:**
  - A) Keras is integrated into TensorFlow, not separate
  - C) TensorBoard handles visualization
  - D) Keras is an API, not a database
</details>

---

### 3. What is the default execution mode in TensorFlow 2.x?

- [ ] A) Graph execution
- [ ] B) Eager execution
- [ ] C) Lazy execution
- [ ] D) Batch execution

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Eager execution

**Explanation:** TensorFlow 2.x defaults to eager execution, where operations execute immediately and return concrete values. This is more intuitive for debugging. Graph execution is available via @tf.function for performance.

- **Why others are wrong:**
  - A) Graph execution was default in TF 1.x
  - C, D) These are not TensorFlow execution modes
</details>

---

### 4. What is a tensor?

- [ ] A) A machine learning algorithm
- [ ] B) A multi-dimensional array of numbers with a uniform data type
- [ ] C) A type of neural network
- [ ] D) A Python function

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) A multi-dimensional array of numbers with a uniform data type

**Explanation:** Tensors generalize scalars (rank 0), vectors (rank 1), and matrices (rank 2) to arbitrary dimensions. They are the fundamental data structure in TensorFlow - all data flows through networks as tensors.

- **Why others are wrong:**
  - A) Tensors are data structures, not algorithms
  - C) Tensors are used BY neural networks, not a type of network
  - D) Tensors are data containers, not functions
</details>

---

### 5. What is the rank of a matrix?

- [ ] A) 0
- [ ] B) 1
- [ ] C) 2
- [ ] D) 3

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) 2

**Explanation:** Rank = number of dimensions. A matrix is a 2D array (rows and columns), so rank = 2. Scalar = rank 0, vector = rank 1, 3D tensor = rank 3.

- **Why others are wrong:**
  - A) Rank 0 is a scalar (single number)
  - B) Rank 1 is a vector (1D array)
  - D) Rank 3 is a 3D tensor
</details>

---

### 6. What does the shape (32, 224, 224, 3) represent for image data?

- [ ] A) 32 channels, 224x224 images, 3 batches
- [ ] B) 32 images, 224x224 pixels, 3 color channels (RGB)
- [ ] C) 224 batches, 32x3 images
- [ ] D) 3 images with 32 features each

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) 32 images, 224x224 pixels, 3 color channels (RGB)

**Explanation:** TensorFlow uses NHWC format: (batch_size, height, width, channels). So (32, 224, 224, 3) = 32 images, each 224 pixels tall and wide, with 3 color channels.

- **Why others are wrong:**
  - A, C, D) These misinterpret the dimension ordering
</details>

---

### 7. What is the purpose of @tf.function decorator?

- [ ] A) To make code run slower
- [ ] B) To convert a Python function into an optimized TensorFlow graph
- [ ] C) To print debug information
- [ ] D) To create new tensors

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) To convert a Python function into an optimized TensorFlow graph

**Explanation:** @tf.function traces the Python function and compiles it into an optimized graph. The first call traces; subsequent calls execute the cached, optimized graph. This can provide significant speedups.

- **Why others are wrong:**
  - A) It makes code run FASTER
  - C) tf.print handles debug printing
  - D) Tensor creation uses tf.constant, tf.Variable, etc.
</details>

---

### 8. What does tf.reshape(tensor, [2, -1]) do?

- [ ] A) Deletes the tensor
- [ ] B) Reshapes to 2 rows, automatically calculating columns
- [ ] C) Creates 2 copies of the tensor
- [ ] D) Raises an error

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Reshapes to 2 rows, automatically calculating columns

**Explanation:** The -1 tells TensorFlow to infer that dimension from the total number of elements. If the tensor has 12 elements, [2, -1] becomes [2, 6] (2 rows x 6 columns = 12).

- **Why others are wrong:**
  - A) reshape doesn't delete data
  - C) reshape reorganizes, doesn't copy
  - D) -1 is valid for one dimension
</details>

---

### 9. What is GradientTape used for in TensorFlow?

- [ ] A) Recording gradients for automatic differentiation
- [ ] B) Playing back audio
- [ ] C) Visualizing the model
- [ ] D) Saving the model to disk

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** A) Recording gradients for automatic differentiation

**Explanation:** GradientTape records operations during the forward pass, then computes gradients during the backward pass. This is essential for training neural networks using gradient descent.

- **Why others are wrong:**
  - B, C, D) These are unrelated to gradient computation
</details>

---

### 10. In Keras Sequential API, what does model.compile() do?

- [ ] A) Executes the model on data
- [ ] B) Configures the model for training with optimizer, loss function, and metrics
- [ ] C) Converts the model to C code
- [ ] D) Prints the model summary

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Configures the model for training with optimizer, loss function, and metrics

**Explanation:** compile() sets up the training configuration: which optimizer to use (Adam, SGD), which loss to minimize (MSE, cross-entropy), and which metrics to track (accuracy). It prepares the model for fit().

- **Why others are wrong:**
  - A) model.fit() executes training
  - C) compile() doesn't generate C code
  - D) model.summary() prints the summary
</details>

---

### 11. What is the correct order of Keras workflow steps?

- [ ] A) compile, build, fit, evaluate
- [ ] B) fit, compile, build, evaluate
- [ ] C) build, compile, fit, evaluate
- [ ] D) evaluate, fit, compile, build

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) build, compile, fit, evaluate

**Explanation:** The typical workflow is: (1) Build the model (Sequential or Functional API), (2) Compile (set optimizer, loss, metrics), (3) Fit (train on data), (4) Evaluate (test performance). Build is often implicit when you add the first layer with input_shape.

- **Why others are wrong:**
  - A, B, D) These have steps out of order - you can't fit before compiling
</details>

---

### 12. What does layers.Dense(64, activation='relu') create?

- [ ] A) 64 separate neural networks
- [ ] B) A fully connected layer with 64 neurons and ReLU activation
- [ ] C) A convolutional layer with 64 filters
- [ ] D) A dropout layer with 64% dropout

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) A fully connected layer with 64 neurons and ReLU activation

**Explanation:** Dense means fully connected - every input connects to every neuron. 64 is the number of neurons (units). activation='relu' applies ReLU after the weighted sum.

- **Why others are wrong:**
  - A) It's one layer, not 64 networks
  - C) Conv2D creates convolutional layers
  - D) Dropout layer uses layers.Dropout()
</details>

---

### 13. What is the formula for parameter count in a Dense layer?

- [ ] A) (input_features + output_neurons)
- [ ] B) (input_features * output_neurons) + output_neurons
- [ ] C) (input_features * output_neurons)
- [ ] D) output_neurons only

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) (input_features * output_neurons) + output_neurons

**Explanation:** Each neuron has: (one weight per input) + (one bias). Total = (input_features x neurons) + neurons. The extra "+ neurons" accounts for biases.

- **Why others are wrong:**
  - A) This is addition, not the correct formula
  - C) This forgets biases
  - D) This forgets weights
</details>

---

### 14. What loss function should you use for multi-class classification with integer labels (not one-hot)?

- [ ] A) binary_crossentropy
- [ ] B) categorical_crossentropy
- [ ] C) sparse_categorical_crossentropy
- [ ] D) mse

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) sparse_categorical_crossentropy

**Explanation:** sparse_categorical_crossentropy takes integer labels (0, 1, 2, ...) directly. categorical_crossentropy requires one-hot encoded labels ([1,0,0], [0,1,0], ...).

- **Why others are wrong:**
  - A) binary_crossentropy is for 2-class problems
  - B) categorical_crossentropy needs one-hot labels
  - D) MSE is for regression
</details>

---

### 15. What does model.fit() return?

- [ ] A) The model weights
- [ ] B) A History object containing training metrics
- [ ] C) The test accuracy
- [ ] D) Nothing

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) A History object containing training metrics

**Explanation:** fit() returns a History object with history.history dictionary containing 'loss', 'accuracy', 'val_loss', 'val_accuracy' (if validation_data provided) for each epoch.

- **Why others are wrong:**
  - A) Weights are stored in the model, accessed via get_weights()
  - C) Test accuracy comes from model.evaluate()
  - D) fit() does return something useful
</details>

---

### 16. Which optimizer is generally the best default choice?

- [ ] A) SGD (Stochastic Gradient Descent)
- [ ] B) Adam
- [ ] C) RMSprop
- [ ] D) Adagrad

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Adam

**Explanation:** Adam (Adaptive Moment Estimation) combines momentum and adaptive learning rates. It typically converges faster and more reliably than vanilla SGD, making it the default choice for most applications.

- **Why others are wrong:**
  - A) SGD works but often requires careful learning rate tuning
  - C) RMSprop is good but Adam usually equals or beats it
  - D) Adagrad can have learning rate issues
</details>

---

## Part 2: True/False

### 17. In TensorFlow, tf.constant creates an immutable tensor while tf.Variable creates a mutable tensor.

- [ ] True
- [ ] False

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** True

**Explanation:** tf.constant creates tensors that cannot be changed. tf.Variable creates tensors that can be updated (via assign, assign_add, etc.). Model weights are stored as Variables so they can be updated during training.
</details>

---

### 18. Broadcasting in TensorFlow allows operations between tensors of different shapes under certain rules.

- [ ] True
- [ ] False

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** True

**Explanation:** Broadcasting automatically expands dimensions to make operations compatible. For example, (3, 4) + (4,) broadcasts the vector to each row of the matrix. Rules: dimensions compared right-to-left, must be equal or one must be 1.
</details>

---

### 19. Keras Sequential API is best for models with multiple inputs or outputs.

- [ ] True
- [ ] False

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** False

**Explanation:** Sequential is for simple stack-of-layers models with single input and single output. For multiple inputs/outputs or complex topologies (branches, skip connections), use the Functional API or Model subclassing.
</details>

---

### 20. model.summary() shows the number of trainable parameters in each layer.

- [ ] True
- [ ] False

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** True

**Explanation:** model.summary() prints a table showing each layer's name, output shape, and parameter count. At the bottom it shows total params, trainable params, and non-trainable params.
</details>

---

### 21. The 'he_normal' initializer is recommended for layers using ReLU activation.

- [ ] True
- [ ] False

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** True

**Explanation:** He initialization (weights ~ Normal(0, sqrt(2/fan_in))) is designed for ReLU, accounting for the fact that half the neurons output 0. It helps maintain signal variance through deep networks.
</details>

---

## Part 3: Code Prediction

### 22. What is the output of this code?

```python
import tensorflow as tf

t = tf.constant([[1, 2, 3], [4, 5, 6]])
print(t.shape)
```

- [ ] A) (2,)
- [ ] B) (3,)
- [ ] C) (2, 3)
- [ ] D) (6,)

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) (2, 3)

**Explanation:** The tensor has 2 rows and 3 columns, making it a 2x3 matrix with shape (2, 3). The outer list has 2 elements, each being a list of 3 elements.
</details>

---

### 23. What does this code print?

```python
import tensorflow as tf

a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])
result = tf.reduce_sum(a * b)
print(result.numpy())
```

- [ ] A) [4, 10, 18]
- [ ] B) 32
- [ ] C) 21
- [ ] D) 6

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) 32

**Explanation:** a * b = [4, 10, 18] (element-wise). tf.reduce_sum adds all elements: 4 + 10 + 18 = 32. This is actually the dot product of a and b.
</details>

---

### 24. What is the shape of the output after this operation?

```python
import tensorflow as tf

x = tf.ones([2, 3, 4])
y = tf.expand_dims(x, axis=0)
print(y.shape)
```

- [ ] A) (2, 3, 4)
- [ ] B) (1, 2, 3, 4)
- [ ] C) (2, 3, 4, 1)
- [ ] D) (6, 4)

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) (1, 2, 3, 4)

**Explanation:** expand_dims adds a dimension of size 1 at the specified axis. axis=0 inserts at the beginning: (2,3,4) becomes (1, 2, 3, 4). This is commonly used to add a batch dimension.
</details>

---

### 25. What will this model.summary() show for the first Dense layer's parameters?

```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(10, activation='softmax')
])
model.summary()
```

- [ ] A) 784
- [ ] B) 128
- [ ] C) 100,352
- [ ] D) 100,480

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** D) 100,480

**Explanation:** Parameters = (input_features x neurons) + biases = (784 x 128) + 128 = 100,352 + 128 = 100,480.
</details>

---

### 26. What does this gradient calculation produce?

```python
import tensorflow as tf

x = tf.Variable(3.0)
with tf.GradientTape() as tape:
    y = x ** 2

grad = tape.gradient(y, x)
print(grad.numpy())
```

- [ ] A) 3.0
- [ ] B) 6.0
- [ ] C) 9.0
- [ ] D) 2.0

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) 6.0

**Explanation:** y = x^2, so dy/dx = 2x. At x=3, the gradient = 2*3 = 6.0. GradientTape automatically computes this derivative.
</details>

---

### 27. What happens when you run this code?

```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(64, activation='relu', input_shape=(10,)),
    Dense(32, activation='relu'),
    Dense(1)  # No activation
])
```

- [ ] A) Error - last layer needs activation
- [ ] B) Creates a model for regression (linear output)
- [ ] C) Creates a model for binary classification
- [ ] D) Error - missing compile

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Creates a model for regression (linear output)

**Explanation:** No activation on the final layer means linear output (any real number), which is correct for regression tasks. Binary classification would use sigmoid; multi-class would use softmax.

- **Why others are wrong:**
  - A) No activation is valid for regression
  - C) Binary classification needs sigmoid
  - D) compile() is separate and optional until training
</details>

---

## Part 4: Fill-in-the-Blank

### 28. TensorFlow uses _______ to compute gradients automatically for backpropagation.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** GradientTape (or automatic differentiation)

**Explanation:** GradientTape records operations as they execute, then computes gradients by traversing the recorded "tape" backward. This automates the calculus needed for training.
</details>

---

### 29. The shape (None, 128) in Keras means the batch size is _______ and each sample has 128 features.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** Variable (or dynamic/unspecified)

**Explanation:** None in shape means that dimension can be any size. For batch dimension, this allows processing any number of samples. The model adapts to whatever batch size you provide.
</details>

---

### 30. To add a Dense layer with 64 neurons and sigmoid activation, you write layers.Dense(64, activation='_______').

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** sigmoid

**Explanation:** The activation parameter takes a string name of the activation function. Common values: 'relu', 'sigmoid', 'softmax', 'tanh'.
</details>

---

### 31. The optimizer _______ is generally recommended as the default choice for training neural networks.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** Adam

**Explanation:** Adam (Adaptive Moment Estimation) combines the benefits of momentum and adaptive learning rates, typically converging faster and more reliably than SGD without much hyperparameter tuning.
</details>

---

### 32. model._______(X_test, y_test) returns the loss and metrics on test data.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** evaluate

**Explanation:** model.evaluate(X, y) computes loss and metrics on provided data without training. It's used to assess final model performance on a held-out test set.
</details>

---

### 33. The TensorFlow tool for visualizing training metrics, model graphs, and more is called _______.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** TensorBoard

**Explanation:** TensorBoard provides interactive visualizations: loss/accuracy plots, model architecture graphs, weight histograms, embedding projectors, and more. Launch with: tensorboard --logdir=logs
</details>

---

## Part 5: Scenario-Based Questions

### 34. You're building a 10-class image classifier. What should the output layer look like?

- [ ] A) Dense(1, activation='sigmoid')
- [ ] B) Dense(10, activation='sigmoid')
- [ ] C) Dense(10, activation='softmax')
- [ ] D) Dense(10, activation='relu')

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) Dense(10, activation='softmax')

**Explanation:** For multi-class classification, you need one output per class (10), and softmax ensures outputs sum to 1 (probability distribution over classes).

- **Why others are wrong:**
  - A) 1 output is for binary classification
  - B) Sigmoid doesn't ensure outputs sum to 1
  - D) ReLU can output any non-negative value, not probabilities
</details>

---

### 35. Your model keeps crashing due to memory errors. What might help?

- [ ] A) Use more layers
- [ ] B) Reduce batch_size in model.fit()
- [ ] C) Add more Dense neurons
- [ ] D) Increase the learning rate

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Reduce batch_size in model.fit()

**Explanation:** Batch size determines how many samples are processed simultaneously, affecting memory usage. Smaller batches use less memory but may train slower or less stably.

- **Why others are wrong:**
  - A, C) More layers/neurons increase memory usage
  - D) Learning rate doesn't affect memory
</details>

---

### 36. You want to prevent your model from overfitting. Which layer could help?

- [ ] A) Dense
- [ ] B) Flatten
- [ ] C) Dropout
- [ ] D) Concatenate

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) Dropout

**Explanation:** Dropout randomly sets a fraction of inputs to 0 during training, preventing the network from relying too heavily on any single feature. This regularization technique reduces overfitting.

- **Why others are wrong:**
  - A) Dense is a computation layer, not regularization
  - B) Flatten reshapes data, doesn't regularize
  - D) Concatenate combines tensors, doesn't regularize
</details>

---

### 37. Your validation accuracy is 95% but test accuracy is 70%. What does this suggest?

- [ ] A) The model is underfitting
- [ ] B) The validation set may not be representative of the test set
- [ ] C) The model is too simple
- [ ] D) Training was too short

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) The validation set may not be representative of the test set

**Explanation:** A large gap between validation and test performance suggests distribution mismatch. The validation set may be too similar to training data, or the test set has different characteristics.

- **Why others are wrong:**
  - A, C, D) These would show poor validation performance too
</details>

---

### 38. You need to process variable-length sequences. Which input_shape specification works?

- [ ] A) input_shape=(100,)
- [ ] B) input_shape=(None, 50)
- [ ] C) input_shape=(50, 100)
- [ ] D) None - Keras can't handle variable lengths

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) input_shape=(None, 50)

**Explanation:** None in input_shape means that dimension can vary. (None, 50) allows variable-length sequences where each timestep has 50 features. This is common for RNNs.

- **Why others are wrong:**
  - A) Fixed 100 features, no sequence dimension
  - C) Fixed 50x100, no variability
  - D) Keras handles variable lengths via None
</details>

---

## Bonus Questions

### 39. What is the difference between model.predict() and model(input)?

- [ ] A) They are exactly the same
- [ ] B) predict() handles batching, callbacks, and is optimized; model(input) is direct call
- [ ] C) model(input) doesn't work
- [ ] D) predict() only works during training

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) predict() handles batching, callbacks, and is optimized; model(input) is direct call

**Explanation:** predict() processes data in batches (default 32), handles progress callbacks, and is optimized for inference. model(input) is a direct forward pass, useful for small inputs or within GradientTape.
</details>

---

### 40. What does tf.squeeze(tensor) do?

- [ ] A) Compresses the tensor values
- [ ] B) Removes dimensions of size 1
- [ ] C) Flattens the tensor
- [ ] D) Normalizes the tensor

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Removes dimensions of size 1

**Explanation:** squeeze removes all dimensions of size 1. For example, shape (1, 3, 1, 4) becomes (3, 4). Optionally specify axis to squeeze only that dimension.

- **Why others are wrong:**
  - A) Values remain unchanged
  - C) Flatten converts to 1D; squeeze removes size-1 dims
  - D) Normalization is different (e.g., batch normalization)
</details>

---

### 41. Why might you use kernel_regularizer=regularizers.l2(0.01) in a Dense layer?

- [ ] A) To speed up training
- [ ] B) To add weight decay and reduce overfitting
- [ ] C) To increase the number of parameters
- [ ] D) To change the activation function

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) To add weight decay and reduce overfitting

**Explanation:** L2 regularization adds a penalty term (0.01 * sum(weights^2)) to the loss, discouraging large weights. This helps prevent overfitting by encouraging simpler models.
</details>

---

### 42. What is "tracing" in the context of @tf.function?

- [ ] A) Debugging print statements
- [ ] B) TensorFlow analyzing the function to build a computation graph
- [ ] C) Tracking model metrics
- [ ] D) Following the data flow visually

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) TensorFlow analyzing the function to build a computation graph

**Explanation:** On first call, @tf.function "traces" the Python function - executes it to see what operations occur - and builds a static graph. Subsequent calls use the cached graph, skipping Python overhead.
</details>

---

*Quiz generated by Practice Quiz Agent for Week 1: AI/ML Fundamentals - Thursday Content*

