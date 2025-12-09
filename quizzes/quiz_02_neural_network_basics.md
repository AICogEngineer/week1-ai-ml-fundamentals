# Weekly Knowledge Check: Neural Network Basics (Wednesday)

## Part 1: Multiple Choice

### 1. What biological structure inspired artificial neural networks?

- [ ] A) The human heart
- [ ] B) The human brain and its neurons
- [ ] C) DNA molecules
- [ ] D) Blood vessels

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) The human brain and its neurons

**Explanation:** Artificial neural networks are inspired by biological neurons that receive signals through dendrites, process them in the cell body, and transmit outputs through axons. The artificial equivalents are inputs, weighted sums, and activations.

- **Why others are wrong:**
  - A, C, D) These are not computational structures that process signals in the way neurons do
</details>

---

### 2. What does an artificial neuron compute?

- [ ] A) Random numbers
- [ ] B) Weighted sum of inputs plus bias, passed through an activation function
- [ ] C) Only the average of inputs
- [ ] D) The maximum input value

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Weighted sum of inputs plus bias, passed through an activation function

**Explanation:** A neuron computes: output = activation(w1*x1 + w2*x2 + ... + wn*xn + b). Each input is multiplied by its weight, summed with the bias, then passed through an activation function.

- **Why others are wrong:**
  - A) Neurons perform deterministic calculations
  - C) Average doesn't use weights
  - D) Maximum would be max pooling, not a basic neuron
</details>

---

### 3. In the perceptron model, what do the weights represent?

- [ ] A) The physical mass of the neuron
- [ ] B) The importance or influence of each input on the output
- [ ] C) The speed of computation
- [ ] D) The number of layers

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) The importance or influence of each input on the output

**Explanation:** Weights determine how much each input contributes to the output. Large positive weights mean the input strongly pushes toward positive class; large negative weights push toward negative class; weights near zero mean the input has little influence.

- **Why others are wrong:**
  - A) Artificial neurons have no physical mass
  - C) Weights don't affect computational speed
  - D) Layer count is a separate architectural choice
</details>

---

### 4. What is the purpose of the bias term (b) in a neuron?

- [ ] A) To make computation faster
- [ ] B) To shift the decision boundary away from the origin
- [ ] C) To remove outliers
- [ ] D) To normalize the inputs

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) To shift the decision boundary away from the origin

**Explanation:** Without bias, the decision boundary (w*x = 0) must pass through the origin. The bias allows the boundary to shift: w*x + b = 0 can be positioned anywhere. This adds flexibility to fit data that isn't centered at zero.

- **Why others are wrong:**
  - A) Bias doesn't affect speed
  - C) Outlier removal is preprocessing
  - D) Normalization is a separate step
</details>

---

### 5. Why was the XOR problem significant in neural network history?

- [ ] A) It was the first problem computers could solve
- [ ] B) It proved that a single perceptron cannot learn non-linearly separable patterns
- [ ] C) It showed neural networks were better than humans
- [ ] D) It demonstrated the first successful AI

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) It proved that a single perceptron cannot learn non-linearly separable patterns

**Explanation:** Minsky and Papert (1969) proved that XOR, which requires a non-linear decision boundary, cannot be solved by a single perceptron. This led to the first "AI winter" but was later resolved by multi-layer networks.

- **Why others are wrong:**
  - A, C, D) These don't describe XOR's historical significance
</details>

---

### 6. What is the primary purpose of activation functions in neural networks?

- [ ] A) To speed up computation
- [ ] B) To introduce non-linearity, enabling the network to learn complex patterns
- [ ] C) To reduce the number of parameters
- [ ] D) To normalize the output

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) To introduce non-linearity, enabling the network to learn complex patterns

**Explanation:** Without activation functions, stacking layers is pointless: multiple linear transformations collapse into a single linear transformation. Non-linear activations break this, allowing each layer to learn distinct representations.

- **Why others are wrong:**
  - A) Some activations (exp, sigmoid) are slower
  - C) Activations don't change parameter count
  - D) Normalization is handled by batch norm or layer norm
</details>

---

### 7. What is the output range of the sigmoid activation function?

- [ ] A) (-infinity, +infinity)
- [ ] B) (-1, 1)
- [ ] C) (0, 1)
- [ ] D) [0, +infinity)

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) (0, 1)

**Explanation:** Sigmoid(z) = 1 / (1 + e^(-z)). As z approaches -infinity, sigmoid approaches 0. As z approaches +infinity, sigmoid approaches 1. The output is strictly between 0 and 1, never reaching either.

- **Why others are wrong:**
  - A) That's the range of linear/no activation
  - B) That's the range of tanh
  - D) That's the range of ReLU
</details>

---

### 8. What is the output range of the tanh activation function?

- [ ] A) (0, 1)
- [ ] B) (-1, 1)
- [ ] C) [0, +infinity)
- [ ] D) (-infinity, +infinity)

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) (-1, 1)

**Explanation:** tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z)). Unlike sigmoid, tanh is zero-centered with outputs ranging from -1 to 1. This can help with gradient flow.

- **Why others are wrong:**
  - A) That's sigmoid's range
  - C) That's ReLU's range
  - D) That's linear/unbounded output
</details>

---

### 9. What is the formula for ReLU?

- [ ] A) f(z) = 1 / (1 + e^(-z))
- [ ] B) f(z) = max(0, z)
- [ ] C) f(z) = (e^z - e^(-z)) / (e^z + e^(-z))
- [ ] D) f(z) = z^2

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) f(z) = max(0, z)

**Explanation:** ReLU (Rectified Linear Unit) outputs the input if positive, otherwise outputs 0. It's computationally cheap, prevents vanishing gradients for positive values, and is the default choice for hidden layers.

- **Why others are wrong:**
  - A) That's the sigmoid formula
  - C) That's the tanh formula
  - D) That's a quadratic function, not an activation
</details>

---

### 10. What is the "dying ReLU" problem?

- [ ] A) ReLU makes networks too slow
- [ ] B) Neurons that output 0 get stuck because their gradient is 0
- [ ] C) ReLU causes memory issues
- [ ] D) ReLU only works for images

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Neurons that output 0 get stuck because their gradient is 0

**Explanation:** When a ReLU neuron always outputs 0 (negative pre-activation), its gradient is 0, so weights never update. The neuron becomes permanently "dead." Leaky ReLU fixes this by allowing a small gradient for negative inputs.

- **Why others are wrong:**
  - A) ReLU is actually very fast
  - C) ReLU has no memory issues
  - D) ReLU works for any data type
</details>

---

### 11. Which activation function allows a small gradient for negative inputs, fixing the dying ReLU problem?

- [ ] A) Sigmoid
- [ ] B) Tanh
- [ ] C) Leaky ReLU
- [ ] D) Step function

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) Leaky ReLU

**Explanation:** Leaky ReLU(z) = z if z > 0, else alpha*z (typically alpha=0.01). The small slope for negative values ensures gradients never become exactly 0, preventing neurons from dying.

- **Why others are wrong:**
  - A, B) These suffer from vanishing gradients at extremes
  - D) Step function has 0 gradient everywhere except the discontinuity
</details>

---

### 12. In a Multi-Layer Perceptron (MLP), what is a "hidden layer"?

- [ ] A) A layer that is invisible to the user
- [ ] B) Any layer between the input and output layers
- [ ] C) The output layer
- [ ] D) A layer that has been removed

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Any layer between the input and output layers

**Explanation:** Hidden layers are intermediate processing layers. They're "hidden" in the sense that we don't directly observe their values - they transform inputs into representations useful for the output layer.

- **Why others are wrong:**
  - A) Hidden layers are fully visible in code
  - C) Output layer is distinct from hidden layers
  - D) Hidden layers are present, not removed
</details>

---

### 13. What does the Universal Approximation Theorem state?

- [ ] A) Neural networks can only approximate linear functions
- [ ] B) A network with one hidden layer and enough neurons can approximate any continuous function
- [ ] C) All neural networks are universal
- [ ] D) Neural networks always find the optimal solution

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) A network with one hidden layer and enough neurons can approximate any continuous function

**Explanation:** The theorem proves that MLPs are universal function approximators - given enough neurons, they can learn any continuous mapping. However, "enough neurons" may be impractically large, which is why depth (more layers) often works better.

- **Why others are wrong:**
  - A) The theorem proves they can approximate non-linear functions
  - C) The theorem has specific conditions (activation type, width)
  - D) The theorem doesn't guarantee training will find that solution
</details>

---

### 14. During forward propagation, data flows:

- [ ] A) From output layer to input layer
- [ ] B) Randomly between layers
- [ ] C) From input layer through hidden layers to output layer
- [ ] D) Only within the hidden layers

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) From input layer through hidden layers to output layer

**Explanation:** Forward propagation computes predictions by passing data forward through the network: Input -> Hidden Layer 1 -> Hidden Layer 2 -> ... -> Output. At each layer: z = Wx + b, then a = activation(z).

- **Why others are wrong:**
  - A) That describes backpropagation (gradient flow)
  - B) Data flow is deterministic, not random
  - D) Data must start at input and end at output
</details>

---

### 15. Which loss function is most appropriate for binary classification?

- [ ] A) Mean Squared Error (MSE)
- [ ] B) Binary Cross-Entropy
- [ ] C) Categorical Cross-Entropy
- [ ] D) Mean Absolute Error (MAE)

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Binary Cross-Entropy

**Explanation:** Binary cross-entropy: L = -[y*log(p) + (1-y)*log(1-p)]. It heavily penalizes confident wrong predictions and is designed for probability outputs from sigmoid activation in binary classification.

- **Why others are wrong:**
  - A, D) MSE and MAE are for regression
  - C) Categorical CE is for multi-class (>2 classes)
</details>

---

### 16. Which loss function is most appropriate for predicting continuous values like house prices?

- [ ] A) Binary Cross-Entropy
- [ ] B) Categorical Cross-Entropy
- [ ] C) Mean Squared Error (MSE)
- [ ] D) Softmax

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) Mean Squared Error (MSE)

**Explanation:** MSE measures the average squared difference between predictions and actual values. It's the standard loss for regression tasks where outputs are continuous numbers.

- **Why others are wrong:**
  - A, B) Cross-entropy losses are for classification
  - D) Softmax is an activation function, not a loss function
</details>

---

## Part 2: True/False

### 17. Without activation functions, a deep neural network with many layers is equivalent to a single linear transformation.

- [ ] True
- [ ] False

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** True

**Explanation:** If all layers are linear (z = Wx + b with no activation), then: z3 = W3(W2(W1*x + b1) + b2) + b3 = (W3*W2*W1)*x + (combined biases) = single linear transformation. Multiple layers collapse to one!
</details>

---

### 18. The perceptron was invented in 2012 as part of the deep learning revolution.

- [ ] True
- [ ] False

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** False

**Explanation:** The perceptron was invented by Frank Rosenblatt in 1958. The 2012 revolution was about deep CNNs (AlexNet) winning ImageNet using GPU training, building on decades of earlier work.
</details>

---

### 19. ReLU is currently the default activation function for hidden layers in most neural networks.

- [ ] True
- [ ] False

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** True

**Explanation:** ReLU is computationally efficient (just max(0, z)), avoids vanishing gradients for positive values, and produces sparse activations. It has been the go-to choice since around 2011-2012.
</details>

---

### 20. In forward propagation, we first apply the activation function, then compute the weighted sum.

- [ ] True
- [ ] False

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** False

**Explanation:** The order is: (1) compute weighted sum z = Wx + b, (2) then apply activation a = activation(z). Activation transforms the linear combination into a non-linear output.
</details>

---

### 21. Cross-entropy loss heavily penalizes confident wrong predictions.

- [ ] True
- [ ] False

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** True

**Explanation:** If actual is 1 and predicted probability is 0.01, cross-entropy loss = -log(0.01) = 4.6. If predicted is 0.99 (correct), loss = -log(0.99) = 0.01. Confident wrong predictions get exponentially higher penalties.
</details>

---

## Part 3: Code Prediction

### 22. What does this code output?

```python
import numpy as np

def relu(z):
    return np.maximum(0, z)

z = np.array([-2, -1, 0, 1, 2])
print(relu(z))
```

- [ ] A) [-2, -1, 0, 1, 2]
- [ ] B) [0, 0, 0, 1, 2]
- [ ] C) [2, 1, 0, 1, 2]
- [ ] D) [0.1, 0.3, 0.5, 0.7, 0.9]

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) [0, 0, 0, 1, 2]

**Explanation:** ReLU(z) = max(0, z). Negative values become 0, non-negative values pass through unchanged. So: max(0,-2)=0, max(0,-1)=0, max(0,0)=0, max(0,1)=1, max(0,2)=2.
</details>

---

### 23. What is the output of this sigmoid calculation?

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

print(round(sigmoid(0), 1))
```

- [ ] A) 0.0
- [ ] B) 0.5
- [ ] C) 1.0
- [ ] D) 0.7

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) 0.5

**Explanation:** sigmoid(0) = 1 / (1 + e^0) = 1 / (1 + 1) = 1/2 = 0.5. The sigmoid function passes through (0, 0.5) - zero input gives 50% probability.
</details>

---

### 24. What is the output shape of this layer operation?

```python
import numpy as np

# Input: 3 features
x = np.array([[1.0, 2.0, 3.0]])  # Shape (1, 3)

# Weight matrix: 3 inputs -> 4 neurons
W = np.random.randn(3, 4)
b = np.zeros(4)

# Forward pass
z = x @ W + b
print(z.shape)
```

- [ ] A) (3,)
- [ ] B) (4,)
- [ ] C) (1, 4)
- [ ] D) (3, 4)

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) (1, 4)

**Explanation:** Matrix multiplication: (1, 3) @ (3, 4) = (1, 4). The batch dimension (1) is preserved, and the output has 4 features (one per neuron in the layer).
</details>

---

### 25. What does this loss calculation compute?

```python
import numpy as np

y_true = np.array([1, 0, 1, 1])
y_pred = np.array([0.9, 0.1, 0.8, 0.7])
epsilon = 1e-15

loss = -np.mean(
    y_true * np.log(y_pred + epsilon) +
    (1 - y_true) * np.log(1 - y_pred + epsilon)
)
print(f"Loss type: ?")
```

- [ ] A) Mean Squared Error
- [ ] B) Binary Cross-Entropy
- [ ] C) Categorical Cross-Entropy
- [ ] D) Hinge Loss

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Binary Cross-Entropy

**Explanation:** The formula -[y*log(p) + (1-y)*log(1-p)] averaged over samples is exactly binary cross-entropy. The epsilon prevents log(0).
</details>

---

### 26. What will be the output of this forward pass?

```python
import numpy as np

def forward_pass(x, W, b):
    z = np.dot(x, W) + b
    return z

x = np.array([2.0, 3.0])
W = np.array([[1.0], [2.0]])
b = np.array([0.5])

output = forward_pass(x, W, b)
print(output)
```

- [ ] A) [8.5]
- [ ] B) [6.0]
- [ ] C) [5.5]
- [ ] D) [3.5]

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** A) [8.5]

**Explanation:** z = dot([2, 3], [[1], [2]]) + [0.5] = (2*1 + 3*2) + 0.5 = (2 + 6) + 0.5 = 8.5
</details>

---

## Part 4: Fill-in-the-Blank

### 27. The three main components of an artificial neuron are inputs, _______, and activation function.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** Weights (and bias)

**Explanation:** A neuron receives inputs, multiplies them by weights, adds a bias, and applies an activation function: output = activation(sum(w_i * x_i) + b).
</details>

---

### 28. The _______ function has outputs between (0, 1) and is commonly used for binary classification output layers.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** Sigmoid

**Explanation:** Sigmoid(z) = 1/(1+e^(-z)) outputs values in (0, 1), interpretable as probabilities. It's the standard output activation for binary classification.
</details>

---

### 29. ReLU stands for _______ Linear Unit.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** Rectified

**Explanation:** ReLU = Rectified Linear Unit. "Rectified" refers to keeping only positive values (like a rectifier in electronics that converts AC to DC by blocking negative values).
</details>

---

### 30. In MLPs, the layers between input and output are called _______ layers.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** Hidden

**Explanation:** Hidden layers process intermediate representations. They're "hidden" because we don't directly observe or specify their outputs - they're learned during training.
</details>

---

### 31. The loss function measures the _______ between predictions and actual values.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** Difference (or error/discrepancy)

**Explanation:** Loss functions quantify how far predictions are from the truth. Lower loss = better predictions. The network learns by minimizing this difference.
</details>

---

### 32. MSE stands for Mean _______ Error.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** Squared

**Explanation:** MSE = Mean Squared Error = average of (actual - predicted)^2. Squaring ensures all errors are positive and penalizes larger errors more.
</details>

---

## Part 5: Scenario-Based Questions

### 33. You're building a neural network for image classification. For hidden layers, which activation function is the best default choice?

- [ ] A) Sigmoid
- [ ] B) Step function
- [ ] C) ReLU
- [ ] D) Linear (no activation)

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) ReLU

**Explanation:** ReLU is the modern default for hidden layers because: (1) computationally efficient, (2) no vanishing gradient for positive values, (3) produces sparse activations, (4) empirically works well.

- **Why others are wrong:**
  - A) Sigmoid has vanishing gradient problems in deep networks
  - B) Step function is non-differentiable, can't train with gradient descent
  - D) No activation collapses the network to a single linear layer
</details>

---

### 34. Your model has sigmoid output with 10 neurons for a 10-class classification problem. What's wrong with this approach?

- [ ] A) Nothing is wrong
- [ ] B) Sigmoid outputs don't sum to 1, should use softmax instead
- [ ] C) Need more than 10 neurons
- [ ] D) Should use ReLU instead

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Sigmoid outputs don't sum to 1, should use softmax instead

**Explanation:** For multi-class classification, outputs should be probabilities that sum to 1 (one class must be chosen). Softmax ensures this: softmax(z_i) = e^z_i / sum(e^z_j). Independent sigmoids could sum to 7.3 or 0.2.
</details>

---

### 35. During training, you notice that many neurons in your network consistently output exactly 0. What might be the problem?

- [ ] A) The model is converging correctly
- [ ] B) Dying ReLU - neurons have negative pre-activations and zero gradients
- [ ] C) The learning rate is too low
- [ ] D) The dataset is too small

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Dying ReLU - neurons have negative pre-activations and zero gradients

**Explanation:** Dying ReLU occurs when neurons consistently receive negative inputs, outputting 0 with gradient 0. They never recover because gradients don't flow. Solutions: Leaky ReLU, better initialization, lower learning rate.
</details>

---

### 36. You need a network output in the range [-1, 1] for a regression task. Which output activation is most appropriate?

- [ ] A) ReLU
- [ ] B) Sigmoid
- [ ] C) Tanh
- [ ] D) Softmax

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) Tanh

**Explanation:** Tanh naturally outputs values in (-1, 1), perfectly matching the required range. Sigmoid would give (0, 1), ReLU gives [0, inf), and softmax gives probabilities summing to 1.
</details>

---

### 37. Your network has 784 inputs (28x28 image) and you want a hidden layer with 256 neurons. How many parameters does this hidden layer have (including biases)?

- [ ] A) 784 * 256 = 200,704
- [ ] B) 784 * 256 + 256 = 200,960
- [ ] C) 784 + 256 = 1,040
- [ ] D) 256 * 256 + 256 = 65,792

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) 784 * 256 + 256 = 200,960

**Explanation:** Parameters = (input_features x neurons) + biases = (784 x 256) + 256 = 200,704 + 256 = 200,960. Each neuron has 784 weights (one per input) plus 1 bias.
</details>

---

### 38. Why would you choose a deep network (many layers) over a wide network (many neurons in one layer)?

- [ ] A) Deep networks are always faster
- [ ] B) Deep networks can learn hierarchical features more efficiently
- [ ] C) Wide networks can't learn non-linear patterns
- [ ] D) Deep networks always have fewer parameters

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Deep networks can learn hierarchical features more efficiently

**Explanation:** Deep networks build abstractions: early layers learn simple features (edges), middle layers combine them (shapes), later layers form complex concepts (objects). This hierarchical learning is more parameter-efficient than trying to learn everything in one wide layer.

- **Why others are wrong:**
  - A) Deep networks can be slower (more sequential operations)
  - C) Universal Approximation says wide networks CAN learn any function
  - D) Parameter count depends on specific architecture
</details>

---

## Bonus Questions

### 39. What event caused the "First AI Winter" and what was the solution?

- [ ] A) Computers were too slow; faster processors solved it
- [ ] B) Perceptrons couldn't solve XOR; multi-layer networks solved it
- [ ] C) Not enough data; the internet provided more data
- [ ] D) Neural networks were forgotten; a new algorithm was invented

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Perceptrons couldn't solve XOR; multi-layer networks solved it

**Explanation:** Minsky and Papert (1969) proved single perceptrons can't solve non-linearly separable problems like XOR. Funding collapsed. The solution came with backpropagation (1986) enabling training of multi-layer networks that CAN solve XOR.
</details>

---

### 40. Why do we need the derivative of the activation function during training?

- [ ] A) To compute the forward pass
- [ ] B) To calculate gradients for backpropagation and weight updates
- [ ] C) To normalize the inputs
- [ ] D) To speed up inference

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) To calculate gradients for backpropagation and weight updates

**Explanation:** Backpropagation uses the chain rule: dLoss/dWeight = dLoss/dOutput * dOutput/dActivation * dActivation/dInput * ... The derivative of the activation function is essential for this gradient computation.
</details>

---

### 41. What is "representation learning" in the context of neural networks?

- [ ] A) The network learns to represent itself graphically
- [ ] B) The network automatically discovers useful features from raw data
- [ ] C) The network memorizes the training data
- [ ] D) The network learns to write code

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) The network automatically discovers useful features from raw data

**Explanation:** Unlike traditional ML where humans engineer features, neural networks learn representations automatically. Hidden layers transform raw inputs (pixels) into increasingly abstract features (edges, textures, parts, objects).
</details>

---

### 42. What is the gradient of ReLU for a positive input (z > 0)?

- [ ] A) 0
- [ ] B) 1
- [ ] C) z
- [ ] D) -1

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) 1

**Explanation:** ReLU(z) = z for z > 0, so d(ReLU)/dz = 1 for z > 0. For z < 0, ReLU(z) = 0, so gradient = 0. At z = 0, the gradient is technically undefined but usually set to 0 or 1.
</details>

---

*Quiz generated by Practice Quiz Agent for Week 1: AI/ML Fundamentals - Wednesday Content*

