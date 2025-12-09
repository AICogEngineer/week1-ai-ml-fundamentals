# Interview Questions: Week 1 - AI/ML Fundamentals

> **Purpose:** This question bank prepares trainees for technical interviews on foundational AI/ML concepts. Questions are categorized by difficulty following the 70-25-5 distribution.
>
> **Self-Quiz Instructions:** Read the question, think of your answer including the **Keywords**, then reveal the solution to compare.

---

## Beginner (Foundational) - 70%

### Q1: What is machine learning, and how does it differ from traditional programming?

**Keywords:** Data, Patterns, Rules, Examples, Model

<details>
<summary>Click to Reveal Answer</summary>

Machine learning is a subset of AI where computers learn patterns from data without being explicitly programmed. 

**Traditional Programming:** Input Data + Rules --> Output
**Machine Learning:** Input Data + Output --> Rules (Model)

In traditional programming, you write explicit rules. In ML, the algorithm discovers rules from examples. The key insight is that ML extracts rules from data rather than having humans encode them.
</details>

---

### Q2: Explain the difference between supervised and unsupervised learning.

**Keywords:** Labels, Prediction, Patterns, Classification/Regression, Clustering

<details>
<summary>Click to Reveal Answer</summary>

**Supervised Learning:**
- Uses **labeled data** (input-output pairs)
- Goal: Learn to **predict** outputs for new inputs
- Examples: Classification (spam detection), Regression (price prediction)

**Unsupervised Learning:**
- Uses **unlabeled data** (inputs only)
- Goal: **Discover patterns** or structure in data
- Examples: Clustering (customer segmentation), Dimensionality reduction

The key distinction is whether you have known correct answers (labels) to learn from.
</details>

---

### Q3: What is the difference between regression and classification?

**Keywords:** Continuous, Discrete, Categories, Numerical, Prediction

<details>
<summary>Click to Reveal Answer</summary>

**Regression:** Predicts **continuous numerical values**
- Examples: House prices ($247,500), Temperature (72.5 degrees), Stock prices
- Output can be any number on a continuous scale

**Classification:** Predicts **discrete categories**
- Examples: Spam/Not Spam, Cat/Dog/Bird, Positive/Negative/Neutral
- Output is one of a predefined set of classes

Use regression when the answer is "how much" and classification when the answer is "which category."
</details>

---

### Q4: What is the purpose of a cost function (loss function) in machine learning?

**Keywords:** Error, Predictions, Optimization, Minimize, Training

<details>
<summary>Click to Reveal Answer</summary>

A cost function measures the **error** between the model's predictions and actual values. It provides the signal that guides learning.

- **Lower cost** = predictions closer to actual = better model
- **Training goal:** Minimize the cost function
- The cost function must be differentiable for gradient-based optimization

Common cost functions: MSE (regression), Cross-Entropy (classification).
</details>

---

### Q5: What are the three main components of an artificial neuron (perceptron)?

**Keywords:** Inputs, Weights, Bias, Activation, Weighted Sum

<details>
<summary>Click to Reveal Answer</summary>

An artificial neuron computes: **output = activation(weighted_sum + bias)**

1. **Inputs & Weights:** Each input is multiplied by a learned weight
2. **Bias:** An offset term that shifts the decision boundary
3. **Activation Function:** Introduces non-linearity (e.g., ReLU, Sigmoid)

The formula: a = activation(w1*x1 + w2*x2 + ... + wn*xn + b)
</details>

---

### Q6: Why do neural networks need activation functions?

**Keywords:** Non-linearity, Linear, Complex Patterns, Collapse, Universal Approximation

<details>
<summary>Click to Reveal Answer</summary>

Without activation functions, neural networks can only learn **linear relationships**, regardless of how many layers they have. Multiple linear transformations collapse into a single linear transformation.

Activation functions introduce **non-linearity**, enabling networks to:
- Learn complex, non-linear patterns
- Act as universal function approximators
- Have each layer learn distinct representations

Common activations: ReLU (hidden layers), Sigmoid/Softmax (output layers).
</details>

---

### Q7: What is ReLU and why is it popular for hidden layers?

**Keywords:** max(0, x), Non-linearity, Vanishing Gradient, Computation, Sparse

<details>
<summary>Click to Reveal Answer</summary>

**ReLU (Rectified Linear Unit):** f(x) = max(0, x)
- Outputs x if positive, 0 otherwise

**Why it's popular:**
1. **Computationally efficient** - just a threshold operation
2. **Avoids vanishing gradients** for positive values (gradient = 1)
3. **Produces sparse activations** - many neurons output 0
4. **Empirically works well** - default choice since ~2012

Limitation: "Dying ReLU" - neurons stuck at 0. Solution: Leaky ReLU.
</details>

---

### Q8: What is a tensor and what are its key properties?

**Keywords:** Multi-dimensional Array, Shape, Rank, Dtype, Dimensions

<details>
<summary>Click to Reveal Answer</summary>

A **tensor** is a multi-dimensional array of numbers with a uniform data type.

**Key properties:**
1. **Shape:** Dimensions of the tensor, e.g., (32, 224, 224, 3)
2. **Rank:** Number of dimensions (scalar=0, vector=1, matrix=2)
3. **Dtype:** Data type (float32, int64, etc.)

Tensors generalize scalars, vectors, and matrices to arbitrary dimensions and are the fundamental data structure in TensorFlow.
</details>

---

### Q9: What is the Keras Sequential API used for?

**Keywords:** Linear Stack, Layers, High-level, Model Building, Simple

<details>
<summary>Click to Reveal Answer</summary>

The **Keras Sequential API** is used to build neural networks as a **linear stack of layers** - where data flows straight through from input to output.

```python
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(10, activation='softmax')
])
```

Best for simple architectures with:
- Single input, single output
- No branches or skip connections

For complex topologies, use the Functional API or Model subclassing.
</details>

---

### Q10: What is a convolutional layer and what does it do?

**Keywords:** Filter, Sliding Window, Feature Detection, Weight Sharing, Spatial

<details>
<summary>Click to Reveal Answer</summary>

A **convolutional layer** applies learnable filters (kernels) that slide across the input to detect features.

**How it works:**
1. A small filter (e.g., 3x3) slides across the image
2. At each position, computes element-wise multiplication and sum
3. Produces a **feature map** showing where features were detected

**Key benefits:**
- **Local connectivity:** Each neuron only sees a small region
- **Weight sharing:** Same filter applied everywhere (translation invariance)
- **Fewer parameters** than fully connected layers
</details>

---

### Q11: What is the purpose of pooling layers in CNNs?

**Keywords:** Dimensionality Reduction, Spatial Invariance, Max/Average, Downsampling

<details>
<summary>Click to Reveal Answer</summary>

**Pooling layers** reduce spatial dimensions while preserving important information.

**Max Pooling:** Takes the maximum value in each region
- Preserves strongest activations (detected features)
- Provides translation invariance within the pooling region

**Average Pooling:** Takes the mean value in each region
- Preserves overall intensity information

A 2x2 pool with stride 2 halves each spatial dimension (28x28 becomes 14x14).
**Note:** Pooling has no learnable parameters.
</details>

---

### Q12: What do training loss curves tell you about model performance?

**Keywords:** Overfitting, Underfitting, Validation, Divergence, Convergence

<details>
<summary>Click to Reveal Answer</summary>

Training curves reveal learning behavior:

**Healthy Learning:**
- Both training and validation loss decrease together
- Small gap between curves

**Overfitting:**
- Training loss decreases while validation loss increases
- Growing gap between curves
- Model memorizes training data

**Underfitting:**
- Both losses plateau at high values
- Model too simple to learn patterns

**Solution:** Monitor validation loss, use early stopping when it stops improving.
</details>

---

## Intermediate (Application/Scenario) - 25%

### Q13: You have customer data but no predefined segments. How would you approach finding natural customer groups?

**Keywords:** Unsupervised, K-Means, Clustering, Elbow Method, Feature Scaling

**Hint:** Think about what type of learning problem this is and what algorithm fits.

<details>
<summary>Click to Reveal Answer</summary>

This is an **unsupervised learning** problem since there are no predefined labels. Use **clustering**, specifically K-Means:

1. **Prepare data:** Select relevant features, scale them (StandardScaler) since K-Means uses distance
2. **Determine K:** Use the **Elbow Method** - plot WCSS vs K, choose the "elbow" point
3. **Apply K-Means:** Initialize centroids, iterate assignment and update until convergence
4. **Interpret clusters:** Analyze centroid characteristics to label segments (e.g., "High-value customers")

Important: Always scale features before distance-based algorithms.
</details>

---

### Q14: Your binary classification model achieves 99% accuracy on a fraud detection task where 99% of transactions are legitimate. Is this good?

**Keywords:** Class Imbalance, Precision, Recall, F1-Score, Baseline

**Hint:** What would happen if the model just predicted "legitimate" for everything?

<details>
<summary>Click to Reveal Answer</summary>

**No, 99% accuracy is misleading here.**

A model predicting "legitimate" for everything achieves 99% accuracy but catches **zero fraud** - useless for the actual goal!

**Better evaluation for imbalanced data:**
- **Precision:** Of predicted frauds, how many were actually fraud?
- **Recall:** Of actual frauds, how many did we catch?
- **F1-Score:** Harmonic mean of precision and recall

For fraud detection, **recall is often more important** - missing fraud is costly.

**Solutions:** Use balanced metrics, adjust classification threshold, oversample minority class (SMOTE).
</details>

---

### Q15: When would you use Manhattan distance instead of Euclidean distance?

**Keywords:** High-dimensional, Sparse, L1, L2, Squaring, Outliers

**Hint:** Think about how squaring affects large differences.

<details>
<summary>Click to Reveal Answer</summary>

**Use Manhattan distance when:**
1. **High-dimensional data:** In high dimensions, Euclidean distance becomes less meaningful (curse of dimensionality). Manhattan is more robust.
2. **Sparse data:** Text analysis, where most features are zero
3. **Outlier sensitivity matters:** Euclidean squares differences, so one large difference dominates. Manhattan treats all dimensions more equally.

**Example:**
- Points A=[1,1,1,1] and B=[1,1,1,5]
- Euclidean focuses on the big difference (4 in last dim)
- Manhattan treats it as one of four equal contributions

Manhattan is common for text/document similarity and recommendation systems.
</details>

---

### Q16: Your neural network's validation accuracy plateaus at 50% for a 2-class problem. What might be wrong and how would you debug it?

**Keywords:** Random Chance, Learning Rate, Architecture, Data Pipeline, Labels

**Hint:** What's the accuracy of random guessing for binary classification?

<details>
<summary>Click to Reveal Answer</summary>

50% accuracy for binary classification is **random chance** - the model isn't learning.

**Debugging steps:**
1. **Check data pipeline:** Are features and labels correctly aligned? Is data shuffled properly?
2. **Verify labels:** Are they correctly encoded (0/1)? Any label leakage?
3. **Learning rate:** Too high (oscillating) or too low (not learning)?
4. **Architecture:** Is the model appropriate for the data?
5. **Loss function:** Using binary_crossentropy for binary classification?
6. **Gradient flow:** Check for vanishing gradients (add batch normalization, use ReLU)

Start simple: Can the model overfit a tiny subset? If not, there's a fundamental issue.
</details>

---

### Q17: How would you choose between using Flatten vs GlobalAveragePooling2D before your classification head?

**Keywords:** Parameters, Overfitting, Spatial Information, Modern Architectures

**Hint:** Consider the number of parameters each approach creates.

<details>
<summary>Click to Reveal Answer</summary>

**Flatten:**
- Converts (H, W, C) to (H*W*C) - preserves all spatial information
- Creates MANY parameters: (7,7,512) flattened = 25,088 features
- Higher overfitting risk
- Use for: small images, when position matters, traditional architectures

**GlobalAveragePooling2D:**
- Converts (H, W, C) to (C) - one value per channel
- Creates FEW parameters: (7,7,512) becomes just 512 features
- Lower overfitting risk, provides spatial invariance
- Use for: large images, modern architectures (ResNet, EfficientNet), transfer learning

**Recommendation:** Start with GlobalAveragePooling2D for most modern use cases - it's more parameter-efficient and often performs as well or better.
</details>

---

### Q18: Your CNN training shows training loss decreasing but validation loss increasing after epoch 5. What would you do?

**Keywords:** Overfitting, Early Stopping, Regularization, Dropout, Data Augmentation

**Hint:** The diverging curves indicate a specific problem.

<details>
<summary>Click to Reveal Answer</summary>

This is classic **overfitting** - the model memorizes training data after epoch 5.

**Immediate actions:**
1. **Early stopping:** Stop training when val_loss stops improving (patience=5)
2. **Restore best weights:** Use the model from epoch 5

**Preventive solutions:**
1. **Dropout:** Add Dropout(0.3-0.5) layers
2. **Data augmentation:** Random flips, rotations, crops
3. **L2 regularization:** kernel_regularizer=l2(0.01)
4. **Reduce model complexity:** Fewer layers or neurons
5. **Get more data:** If possible

**Code:**
```python
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
```
</details>

---

## Advanced (Deep Dive/System Design) - 5%

### Q19: Explain what happens "under the hood" when you call model.fit() in Keras.

**Keywords:** Forward Pass, Backpropagation, Gradient Descent, Batch Processing, Epochs

<details>
<summary>Click to Reveal Answer</summary>

When you call `model.fit(X, y, epochs=10, batch_size=32)`:

**For each epoch:**
1. **Shuffle** the training data (if shuffle=True)
2. **Divide** into batches of 32 samples

**For each batch:**
3. **Forward pass:** Data flows through layers, computing predictions
4. **Loss calculation:** Compare predictions to actual labels
5. **Backpropagation:** Compute gradients of loss w.r.t. each weight using chain rule (GradientTape)
6. **Weight update:** Optimizer (e.g., Adam) adjusts weights: w = w - learning_rate * gradient

**After each epoch:**
7. **Validation:** Forward pass on validation data (no gradient computation)
8. **Callbacks:** EarlyStopping, ModelCheckpoint, etc. execute
9. **History update:** Record loss/metrics

**Key:** Backpropagation uses automatic differentiation - TensorFlow builds a computation graph and traverses it backward to compute all gradients efficiently.
</details>

---

### Q20: Why can a single perceptron not solve the XOR problem, and how do multi-layer networks solve it?

**Keywords:** Linear Separability, Decision Boundary, Hidden Layer, Feature Transformation, Non-linear

<details>
<summary>Click to Reveal Answer</summary>

**Why single perceptron fails:**
A perceptron creates a **linear decision boundary** (a straight line in 2D). XOR's truth table:
```
(0,0)->0  (0,1)->1  (1,0)->1  (1,1)->0
```
No straight line can separate the 0s from the 1s - they're diagonally opposite.

**How MLPs solve it:**
The hidden layer **transforms the input space** into a new representation where classes become linearly separable.

**Intuition:** Hidden neurons learn intermediate features:
- h1 = OR(x1, x2) - fires if either input is 1
- h2 = NAND(x1, x2) - fires unless both inputs are 1

In (h1, h2) space, XOR becomes: AND(h1, h2) - now linearly separable!

**Key insight:** Each layer transforms data into representations more suitable for the next layer's task. This is representation learning.
</details>

---

### Q21: Design a CNN architecture for classifying 64x64 RGB images into 100 categories. Justify your design choices.

**Keywords:** Conv Blocks, Pooling, Feature Maps, Flatten/GAP, Softmax, Parameter Count

<details>
<summary>Click to Reveal Answer</summary>

**Proposed Architecture:**

```python
model = Sequential([
    # Block 1: 64x64x3 -> 32x32x32
    Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(64,64,3)),
    Conv2D(32, (3,3), activation='relu', padding='same'),
    MaxPooling2D((2,2)),
    
    # Block 2: 32x32x32 -> 16x16x64
    Conv2D(64, (3,3), activation='relu', padding='same'),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    MaxPooling2D((2,2)),
    
    # Block 3: 16x16x64 -> 8x8x128
    Conv2D(128, (3,3), activation='relu', padding='same'),
    Conv2D(128, (3,3), activation='relu', padding='same'),
    MaxPooling2D((2,2)),
    
    # Classification Head
    GlobalAveragePooling2D(),  # 8x8x128 -> 128
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(100, activation='softmax')
])
```

**Design Justifications:**
1. **Stacked Conv layers before pooling:** Increases receptive field, learns complex features before downsampling (VGGNet pattern)
2. **Increasing filter count (32->64->128):** Early layers need fewer filters (edges), deeper layers need more (complex patterns)
3. **padding='same':** Preserves spatial dimensions within blocks for easier reasoning
4. **GlobalAveragePooling2D:** Drastically reduces parameters vs Flatten (128 vs 8192 features)
5. **Dropout(0.5):** Regularization for 100-class problem with potential overfitting
6. **Softmax with 100 units:** One probability per class, summing to 1

**Approximate parameters:** ~300K (much less than equivalent Dense network)
</details>

---

## Bonus Interview Scenarios

### Q22: Walk me through how you would approach a new ML project from data to deployment.

**Keywords:** Problem Definition, Data Exploration, Baseline, Iteration, Evaluation, Deployment

<details>
<summary>Click to Reveal Answer</summary>

**1. Problem Definition**
- Understand business goal, define success metrics
- Determine: Classification vs Regression? Supervised vs Unsupervised?

**2. Data Understanding**
- Explore data: distributions, missing values, outliers
- Understand features, identify target variable
- Check for class imbalance

**3. Data Preparation**
- Clean data, handle missing values
- Feature engineering, scaling
- Train/validation/test split

**4. Establish Baseline**
- Simple model (logistic regression, decision tree)
- Understand what "good" looks like

**5. Model Development**
- Start simple, increase complexity as needed
- Hyperparameter tuning
- Cross-validation for robust estimates

**6. Evaluation**
- Appropriate metrics (accuracy, F1, MSE depending on task)
- Confusion matrix, learning curves
- Test set evaluation (only once!)

**7. Deployment & Monitoring**
- Model serialization
- API/service deployment
- Monitor for data drift, performance degradation
</details>

---

### Q23: Explain the bias-variance tradeoff in the context of neural networks.

**Keywords:** Underfitting, Overfitting, Generalization, Model Complexity, Regularization

<details>
<summary>Click to Reveal Answer</summary>

**Bias-Variance Tradeoff:**

**High Bias (Underfitting):**
- Model too simple to capture patterns
- High training AND validation error
- Neural network: too few layers/neurons

**High Variance (Overfitting):**
- Model memorizes training data, fails on new data
- Low training error, high validation error
- Neural network: too many parameters, trained too long

**The Tradeoff:**
- Increasing model complexity reduces bias but increases variance
- Goal: Find the sweet spot with good generalization

**Neural Network Solutions:**
- **For high bias:** More layers, more neurons, train longer
- **For high variance:** Regularization (L2, Dropout), early stopping, data augmentation, more data

**Modern insight:** Very large neural networks, when properly regularized, can achieve low bias AND low variance ("double descent").
</details>

---

*Interview Question Bank generated by Quality Assurance Agent for Week 1: AI/ML Fundamentals*
*Distribution: 12 Beginner (52%), 6 Intermediate (26%), 3 Advanced (13%), 2 Bonus (9%)*

