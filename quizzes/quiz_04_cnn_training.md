# Weekly Knowledge Check: CNN & Training (Friday)

## Part 1: Multiple Choice

### 1. What is the main advantage of CNNs over Dense networks for image tasks?

- [ ] A) CNNs are faster to train
- [ ] B) CNNs exploit spatial structure through local connectivity and weight sharing
- [ ] C) CNNs can only process images
- [ ] D) CNNs have more parameters

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) CNNs exploit spatial structure through local connectivity and weight sharing

**Explanation:** CNNs use local filters that detect patterns anywhere in the image (translation invariance), share weights across locations (fewer parameters), and learn hierarchical features (edges to textures to objects).

- **Why others are wrong:**
  - A) CNNs can actually be slower due to many convolutions
  - C) CNNs also work on audio, time series, text
  - D) CNNs typically have FEWER parameters than equivalent Dense networks
</details>

---

### 2. What does "translation invariance" mean in CNNs?

- [ ] A) The network can translate text
- [ ] B) A feature can be detected regardless of its position in the image
- [ ] C) The network moves during training
- [ ] D) Images are automatically translated to different languages

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) A feature can be detected regardless of its position in the image

**Explanation:** The same convolutional filter slides across the entire image, so it detects the same pattern (e.g., a cat's eye) whether it appears in the top-left corner or bottom-right corner.

- **Why others are wrong:**
  - A, D) "Translation" here refers to spatial position, not language
  - C) Networks don't physically move
</details>

---

### 3. In a convolution operation, what does the filter/kernel do?

- [ ] A) Deletes pixels from the image
- [ ] B) Slides across the input, computing weighted sums to detect features
- [ ] C) Increases the image size
- [ ] D) Converts the image to grayscale

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Slides across the input, computing weighted sums to detect features

**Explanation:** The filter is a small matrix (e.g., 3x3) that slides across the image. At each position, it computes the element-wise product and sum, producing a value in the output feature map. Different filters detect different features.

- **Why others are wrong:**
  - A) Convolution doesn't delete data
  - C) Convolution typically reduces or maintains size
  - D) Color conversion is separate preprocessing
</details>

---

### 4. What is a feature map?

- [ ] A) A geographic map of features
- [ ] B) The output of a convolutional layer showing where features were detected
- [ ] C) A list of all possible features
- [ ] D) The input image

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) The output of a convolutional layer showing where features were detected

**Explanation:** Each filter produces one feature map. High values indicate where the filter's feature was detected. With 32 filters, you get 32 feature maps, each highlighting different patterns.

- **Why others are wrong:**
  - A) Not a geographic map
  - C) Feature maps are 2D arrays of activations
  - D) Feature maps are outputs, not inputs
</details>

---

### 5. What is the output size formula for convolution with padding='valid'?

- [ ] A) Output = Input + Filter
- [ ] B) Output = (Input - Filter) / Stride + 1
- [ ] C) Output = Input * Filter
- [ ] D) Output = Input

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Output = (Input - Filter) / Stride + 1

**Explanation:** With valid padding (no padding), the filter can't extend past the edges. For input=28, filter=3, stride=1: Output = (28-3)/1 + 1 = 26. The image shrinks.

- **Why others are wrong:**
  - A, C) These would increase size
  - D) That would be padding='same'
</details>

---

### 6. If input image is 28x28x1 and you apply Conv2D with 32 filters of 3x3 and valid padding, what is the output shape?

- [ ] A) (28, 28, 32)
- [ ] B) (26, 26, 32)
- [ ] C) (28, 28, 1)
- [ ] D) (26, 26, 1)

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) (26, 26, 32)

**Explanation:** Spatial: (28-3)/1 + 1 = 26x26. Channels: 32 filters produce 32 feature maps. Final shape: (26, 26, 32).

- **Why others are wrong:**
  - A) Valid padding reduces spatial dimensions
  - C, D) 32 filters produce 32 channels, not 1
</details>

---

### 7. How many parameters does Conv2D(32, (3,3)) have when the input has 1 channel?

- [ ] A) 32
- [ ] B) 288
- [ ] C) 320
- [ ] D) 9

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) 320

**Explanation:** Parameters = (kernel_h x kernel_w x input_channels + 1) x num_filters = (3 x 3 x 1 + 1) x 32 = (9 + 1) x 32 = 10 x 32 = 320. The +1 is for bias per filter.

- **Why others are wrong:**
  - A) 32 is just the number of filters
  - B) 288 forgets the biases
  - D) 9 is just kernel size, ignoring filters and biases
</details>

---

### 8. What does Max Pooling 2x2 with stride 2 do to spatial dimensions?

- [ ] A) Doubles the dimensions
- [ ] B) Halves the dimensions
- [ ] C) Keeps dimensions the same
- [ ] D) Reduces to 1x1

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Halves the dimensions

**Explanation:** A 2x2 pool with stride 2 takes the maximum from each non-overlapping 2x2 region. This reduces each spatial dimension by half: 28x28 becomes 14x14.

- **Why others are wrong:**
  - A) Pooling reduces, not increases
  - C) 2x2 pooling does change dimensions
  - D) Multiple pooling layers might eventually reach 1x1
</details>

---

### 9. What is the main difference between Max Pooling and Average Pooling?

- [ ] A) Max Pooling is faster
- [ ] B) Max Pooling takes the maximum value; Average Pooling takes the mean
- [ ] C) Average Pooling only works on 3D data
- [ ] D) Max Pooling increases dimensions

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Max Pooling takes the maximum value; Average Pooling takes the mean

**Explanation:** Max Pooling preserves the strongest activation (detected feature) in each region. Average Pooling preserves the overall intensity. Max Pooling is more common for feature detection.

- **Why others are wrong:**
  - A) Speed is similar
  - C) Both work on any dimension
  - D) Both pooling types reduce dimensions
</details>

---

### 10. How many learnable parameters does a MaxPooling2D layer have?

- [ ] A) 4
- [ ] B) Depends on pool size
- [ ] C) 0
- [ ] D) Same as the previous layer

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) 0

**Explanation:** Pooling layers have no learnable parameters - they simply apply a fixed operation (max or average) to regions. No weights or biases to learn.

- **Why others are wrong:**
  - A, B, D) Pooling is a parameter-free operation
</details>

---

### 11. What is the purpose of the Flatten layer in a CNN?

- [ ] A) To make the image flat (2D)
- [ ] B) To convert multi-dimensional feature maps into a 1D vector for Dense layers
- [ ] C) To reduce the number of parameters
- [ ] D) To apply convolution

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) To convert multi-dimensional feature maps into a 1D vector for Dense layers

**Explanation:** Dense layers expect 1D input (batch, features). Flatten reshapes (batch, height, width, channels) to (batch, height*width*channels), connecting the CNN feature extractor to the Dense classification head.

- **Why others are wrong:**
  - A) Images are already 2D; Flatten makes 1D
  - C) Flatten doesn't change parameter count
  - D) Convolution uses Conv2D layers
</details>

---

### 12. What does Global Average Pooling do?

- [ ] A) Averages across the batch dimension
- [ ] B) Reduces each feature map to a single value (its average)
- [ ] C) Increases spatial dimensions
- [ ] D) Applies learned weights

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Reduces each feature map to a single value (its average)

**Explanation:** GlobalAveragePooling2D takes the average of each HxW feature map, producing one value per channel. Shape (batch, 7, 7, 512) becomes (batch, 512). It's an alternative to Flatten + Dense.

- **Why others are wrong:**
  - A) It pools spatially, not across batches
  - C) It drastically reduces dimensions
  - D) No learnable parameters
</details>

---

### 13. In training visualization, what pattern indicates overfitting?

- [ ] A) Both training and validation loss decrease together
- [ ] B) Training loss decreases while validation loss increases
- [ ] C) Both losses remain high
- [ ] D) Both losses oscillate wildly

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Training loss decreases while validation loss increases

**Explanation:** Overfitting means the model memorizes training data but doesn't generalize. Training loss keeps improving, but validation loss worsens - the gap between curves is the tell-tale sign.

- **Why others are wrong:**
  - A) This is healthy learning
  - C) This is underfitting
  - D) This suggests learning rate issues
</details>

---

### 14. What pattern indicates underfitting?

- [ ] A) Validation loss lower than training loss
- [ ] B) Both training and validation loss plateau at high values
- [ ] C) Perfect training accuracy
- [ ] D) Rapidly decreasing loss

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Both training and validation loss plateau at high values

**Explanation:** Underfitting means the model is too simple to learn the patterns. Both losses stay high because the model can't even fit the training data well.

- **Why others are wrong:**
  - A) This is unusual but possible with regularization
  - C) Perfect training accuracy often indicates overfitting
  - D) Rapidly decreasing loss is good learning
</details>

---

### 15. What does EarlyStopping callback do?

- [ ] A) Stops training when validation metric stops improving
- [ ] B) Makes training slower
- [ ] C) Removes layers from the model
- [ ] D) Increases the learning rate

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** A) Stops training when validation metric stops improving

**Explanation:** EarlyStopping monitors a metric (usually val_loss) and stops training after 'patience' epochs without improvement. It prevents overfitting by stopping at the optimal point.

- **Why others are wrong:**
  - B) It can make training finish sooner
  - C) It doesn't change architecture
  - D) Learning rate scheduling is separate
</details>

---

### 16. What does padding='same' do in Conv2D?

- [ ] A) Uses the same filter for all layers
- [ ] B) Pads the input so output has the same spatial dimensions as input
- [ ] C) Makes all values the same
- [ ] D) Copies the input to output

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Pads the input so output has the same spatial dimensions as input

**Explanation:** padding='same' adds zeros around the input edges so that with stride=1, output size equals input size. For a 28x28 input with 3x3 filter, output is still 28x28.

- **Why others are wrong:**
  - A) Each layer has its own filters
  - C, D) Padding preserves size, doesn't copy values
</details>

---

## Part 2: True/False

### 17. CNNs learn hierarchical features: early layers detect simple patterns (edges), later layers detect complex patterns (objects).

- [ ] True
- [ ] False

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** True

**Explanation:** Visualization of CNN layers shows this hierarchy: Layer 1 learns edge detectors, Layer 2 combines them into textures, deeper layers recognize parts (eyes, wheels), and final layers recognize whole objects.
</details>

---

### 18. Pooling layers have learnable parameters that are updated during training.

- [ ] True
- [ ] False

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** False

**Explanation:** Pooling applies fixed operations (max or average) with no learnable parameters. It simply aggregates values in each region - no weights to update.
</details>

---

### 19. A typical CNN architecture follows the pattern: [Conv -> Pool]* -> Flatten -> Dense -> Output.

- [ ] True
- [ ] False

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** True

**Explanation:** This is the standard pattern: alternating Conv and Pool blocks extract and compress features, then Flatten converts to 1D, and Dense layers perform classification. Some variations exist but this is the classic structure.
</details>

---

### 20. Stride=2 in a convolutional layer reduces the spatial dimensions by half.

- [ ] True
- [ ] False

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** True

**Explanation:** With stride=2, the filter moves 2 pixels between positions, so output size is approximately half the input size. This is often used instead of pooling for downsampling in modern architectures.
</details>

---

### 21. When validation loss is decreasing, you should continue training.

- [ ] True
- [ ] False

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** True

**Explanation:** Decreasing validation loss indicates the model is still improving on unseen data. Stop when validation loss stops decreasing (or starts increasing), which signals potential overfitting.
</details>

---

### 22. A 1x1 convolution can be used to change the number of channels without changing spatial dimensions.

- [ ] True
- [ ] False

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** True

**Explanation:** A 1x1 convolution looks at each pixel individually across all channels. With N filters, it transforms C input channels to N output channels. GoogLeNet/Inception uses this for dimensionality reduction.
</details>

---

## Part 3: Code Prediction

### 23. What is the shape after this code?

```python
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf

x = tf.random.normal([1, 28, 28, 1])
x = Conv2D(32, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
print(x.shape)
```

- [ ] A) (1, 28, 28, 32)
- [ ] B) (1, 26, 26, 32)
- [ ] C) (1, 13, 13, 32)
- [ ] D) (1, 14, 14, 32)

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) (1, 13, 13, 32)

**Explanation:** Conv2D with valid padding: (28-3)/1 + 1 = 26. So after Conv: (1, 26, 26, 32). MaxPooling 2x2: 26/2 = 13. Final: (1, 13, 13, 32).
</details>

---

### 24. How many parameters does this model have?

```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(10, activation='softmax')
])
```

- [ ] A) 280
- [ ] B) 784
- [ ] C) 7840
- [ ] D) 7850

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** D) 7850

**Explanation:** Flatten: 0 params (just reshapes). Dense: (784 x 10) + 10 = 7840 + 10 = 7850. The 784 comes from 28*28 flattened inputs.
</details>

---

### 25. What does this pooling operation output?

```python
import numpy as np
from tensorflow.keras.layers import MaxPooling2D
import tensorflow as tf

x = np.array([[
    [[1], [3], [2], [4]],
    [[5], [2], [1], [0]],
    [[2], [8], [3], [1]],
    [[4], [1], [6], [2]]
]], dtype=np.float32)

pool = MaxPooling2D(pool_size=(2, 2), strides=2)
result = pool(x)
print(result[0, :, :, 0])
```

- [ ] A) [[5, 4], [8, 6]]
- [ ] B) [[2.75, 1.75], [3.75, 3]]
- [ ] C) [[1, 2], [2, 1]]
- [ ] D) [[3, 2], [4, 3]]

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** A) [[5, 4], [8, 6]]

**Explanation:** Max pooling takes maximum from each 2x2 region:
- Top-left: max(1,3,5,2) = 5
- Top-right: max(2,4,1,0) = 4
- Bottom-left: max(2,8,4,1) = 8
- Bottom-right: max(3,1,6,2) = 6
</details>

---

### 26. What shape does Flatten produce?

```python
from tensorflow.keras.layers import Flatten
import tensorflow as tf

x = tf.random.normal([16, 7, 7, 64])
flat = Flatten()(x)
print(flat.shape)
```

- [ ] A) (16, 7, 7, 64)
- [ ] B) (16, 3136)
- [ ] C) (7, 7, 64)
- [ ] D) (3136,)

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) (16, 3136)

**Explanation:** Flatten keeps the batch dimension and flattens everything else. 7 * 7 * 64 = 3136. So shape becomes (batch_size, 3136) = (16, 3136).
</details>

---

### 27. What does this training history show?

```python
# After training, history contains:
history.history['loss'] = [2.3, 1.5, 0.8, 0.4, 0.2]
history.history['val_loss'] = [2.2, 1.6, 1.4, 1.5, 1.8]
```

- [ ] A) Good learning with no issues
- [ ] B) Underfitting - model too simple
- [ ] C) Overfitting - training improves but validation worsens
- [ ] D) Learning rate too high

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) Overfitting - training improves but validation worsens

**Explanation:** Training loss steadily decreases (2.3 -> 0.2), but validation loss decreases initially then increases (2.2 -> 1.6 -> 1.4 -> 1.5 -> 1.8). The diverging curves indicate overfitting starting around epoch 3.
</details>

---

## Part 4: Fill-in-the-Blank

### 28. A _______ is a small matrix that slides across the image to detect features in a convolutional layer.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** Filter (or kernel)

**Explanation:** Filters/kernels are the learnable parameters in Conv layers. They detect specific patterns like edges, corners, or textures depending on their learned weights.
</details>

---

### 29. The _______ layer reduces spatial dimensions by taking the maximum (or average) value in each region.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** Pooling

**Explanation:** Pooling (MaxPooling or AveragePooling) downsamples feature maps by aggregating values in each region, reducing computational cost and providing some translation invariance.
</details>

---

### 30. When validation loss increases while training loss decreases, the model is _______.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** Overfitting

**Explanation:** This divergence indicates the model is memorizing training data rather than learning generalizable patterns. Solutions include more data, regularization, or early stopping.
</details>

---

### 31. The _______ layer converts multi-dimensional feature maps to a 1D vector before Dense layers.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** Flatten

**Explanation:** Flatten reshapes (batch, H, W, C) to (batch, H*W*C), creating the 1D feature vector that Dense layers require.
</details>

---

### 32. _______ callbacks in Keras can automatically stop training when a metric stops improving.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** EarlyStopping

**Explanation:** EarlyStopping monitors val_loss (or other metric) and stops after 'patience' epochs without improvement. This prevents overfitting and saves training time.
</details>

---

### 33. The tool for visualizing training metrics, model graphs, and more in TensorFlow is _______.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** TensorBoard

**Explanation:** TensorBoard provides interactive dashboards for loss/accuracy plots, model architecture visualization, weight histograms, and more. Use the TensorBoard callback during training.
</details>

---

## Part 5: Scenario-Based Questions

### 34. You have a 224x224x3 image. Using Dense layers directly would require how many parameters for the first layer with 512 neurons?

- [ ] A) About 25 million
- [ ] B) About 77 million
- [ ] C) About 1 million
- [ ] D) About 512 thousand

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) About 77 million

**Explanation:** Flattened input: 224 * 224 * 3 = 150,528 features. Dense(512) parameters: (150,528 * 512) + 512 = 77,070,848. This is why CNNs with weight sharing are preferred!
</details>

---

### 35. Your CNN training shows both training and validation accuracy stuck at ~10% for a 10-class problem. What is likely happening?

- [ ] A) Overfitting
- [ ] B) Random chance / model not learning
- [ ] C) Perfect performance
- [ ] D) Learning rate is optimal

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Random chance / model not learning

**Explanation:** 10% accuracy on 10 classes is random guessing (1/10 chance). The model isn't learning - possible causes: learning rate too high/low, bug in data pipeline, wrong loss function, vanishing gradients.
</details>

---

### 36. You want to reduce overfitting in your CNN. Which is NOT an effective strategy?

- [ ] A) Add Dropout layers
- [ ] B) Use data augmentation
- [ ] C) Increase model complexity (more layers)
- [ ] D) Apply L2 regularization

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) Increase model complexity (more layers)

**Explanation:** More complexity typically INCREASES overfitting risk. Dropout, data augmentation, and regularization all help REDUCE overfitting by preventing the model from memorizing training data.
</details>

---

### 37. For a small 28x28 grayscale image classification task, which approach is most appropriate?

- [ ] A) Very deep ResNet-152
- [ ] B) Simple CNN with 2-3 conv layers
- [ ] C) Only Dense layers, no convolution
- [ ] D) 3D convolutions

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Simple CNN with 2-3 conv layers

**Explanation:** Small images don't need very deep networks - a few Conv layers suffice for MNIST-like tasks. ResNet-152 would be overkill. Some simple tasks might work with Dense only, but CNN is more appropriate.

- **Why others are wrong:**
  - A) Overkill for simple task, prone to overfitting
  - C) Ignores spatial structure
  - D) 3D convolution is for volumetric data
</details>

---

### 38. Your validation loss hasn't improved for 10 epochs. What should you do?

- [ ] A) Keep training for 100 more epochs
- [ ] B) Stop training - the model has likely converged or is overfitting
- [ ] C) Remove all regularization
- [ ] D) Increase the learning rate 10x

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Stop training - the model has likely converged or is overfitting

**Explanation:** No improvement for 10 epochs suggests the model has reached its capacity on this data/architecture. Continuing wastes time and risks overfitting. EarlyStopping automates this decision.
</details>

---

### 39. What is the benefit of using GlobalAveragePooling2D instead of Flatten before the classification head?

- [ ] A) It increases accuracy
- [ ] B) It significantly reduces parameters while providing some spatial invariance
- [ ] C) It's required for all CNNs
- [ ] D) It makes training faster

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) It significantly reduces parameters while providing some spatial invariance

**Explanation:** GAP reduces (7,7,512) to (512), compared to Flatten's (25088). The subsequent Dense layer has ~50x fewer parameters. GAP also provides translation invariance by averaging spatial information.
</details>

---

## Bonus Questions

### 40. What makes AlexNet (2012) historically significant?

- [ ] A) It was the first neural network ever created
- [ ] B) It won ImageNet by a large margin using GPU-trained deep CNNs
- [ ] C) It used only Dense layers
- [ ] D) It was smaller than all previous models

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) It won ImageNet by a large margin using GPU-trained deep CNNs

**Explanation:** AlexNet achieved 16.4% error vs. 26% for the runner-up, proving that deep CNNs trained on GPUs could dramatically outperform traditional computer vision. This sparked the deep learning revolution.
</details>

---

### 41. What is the purpose of stride in convolution?

- [ ] A) To make filters larger
- [ ] B) To control how many pixels the filter moves between positions
- [ ] C) To add more filters
- [ ] D) To change the activation function

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) To control how many pixels the filter moves between positions

**Explanation:** Stride=1 moves the filter one pixel at a time. Stride=2 moves two pixels, producing roughly half the output size. Larger strides downsample the image.
</details>

---

### 42. Why might you use multiple Conv layers before pooling (e.g., Conv -> Conv -> Pool)?

- [ ] A) It's required by TensorFlow
- [ ] B) It allows learning more complex features before downsampling
- [ ] C) It reduces parameters
- [ ] D) It speeds up training

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) It allows learning more complex features before downsampling

**Explanation:** Stacking Conv layers increases the "receptive field" (how much of the input each output sees) and allows learning more complex feature combinations before pooling loses spatial resolution. VGGNet popularized this pattern.
</details>

---

### 43. What does the ModelCheckpoint callback do?

- [ ] A) Checks if the model is correct
- [ ] B) Saves the model at specified intervals or when metrics improve
- [ ] C) Creates checkpoints in the code
- [ ] D) Validates the model architecture

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Saves the model at specified intervals or when metrics improve

**Explanation:** ModelCheckpoint can save the model every epoch or only when val_loss improves (save_best_only=True). This ensures you keep the best-performing version even if later epochs overfit.
</details>

---

### 44. A filter that looks like [[1,0,-1],[1,0,-1],[1,0,-1]] detects what?

- [ ] A) Horizontal edges
- [ ] B) Vertical edges
- [ ] C) Diagonal edges
- [ ] D) Blurring

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Vertical edges

**Explanation:** This filter has positive values on the left, zeros in the middle, and negative values on the right. It responds strongly to vertical transitions from light to dark (vertical edges).
</details>

---

*Quiz generated by Practice Quiz Agent for Week 1: AI/ML Fundamentals - Friday Content*

---

## Week 1 Summary

Congratulations on completing all four practice quizzes for Week 1: AI/ML Fundamentals!

**Topics Covered:**
- **Tuesday (Quiz 1):** ML algorithms, supervised/unsupervised learning, regression, classification, K-Means, distance metrics
- **Wednesday (Quiz 2):** Neural networks, perceptrons, activation functions, MLPs, forward propagation, loss functions
- **Thursday (Quiz 3):** TensorFlow, tensors, shapes, graphs, Keras Sequential API, Dense layers
- **Friday (Quiz 4):** CNNs, convolution, pooling, flattening, training visualization, overfitting/underfitting

**Key Concepts Mastered:**
1. The difference between supervised and unsupervised learning
2. How neural networks compute outputs through weighted sums and activations
3. Building models with TensorFlow/Keras
4. CNN architecture for image processing
5. Monitoring and diagnosing training

*You are now ready for Week 2's deeper dive into backpropagation and gradient descent!*

