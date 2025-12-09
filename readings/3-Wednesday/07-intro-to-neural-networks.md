# Introduction to Neural Networks

## Learning Objectives

- Understand the biological inspiration behind artificial neural networks
- Explain the concept of an artificial neuron and how it processes information
- Describe how neural networks learn representations from data
- Contextualize neural networks within the historical arc of AI development

## Why This Matters

Yesterday, you learned how traditional machine learning algorithms find patterns in data. Today, we cross a threshold into a fundamentally different paradigm: **neural networks** - computational systems inspired by the human brain.

In our **From Zero to Neural** journey, this is the "neural" part. Neural networks power the AI revolution: they understand your voice commands, generate images from text descriptions, translate languages in real-time, and beat world champions at complex games.

But here's the key insight: neural networks aren't magic. They're mathematical functions composed of simple building blocks. By the end of today, you'll understand exactly how these building blocks work.

## The Concept

### The Biological Inspiration

The human brain contains approximately 86 billion neurons, each connected to thousands of others through synapses. When you recognize a face, remember a fact, or decide to take action, electrical signals flow through networks of neurons.

```
Biological Neuron:

    Dendrites           Cell Body           Axon
    (inputs)            (processing)        (output)
    
       \   |   /            O               ------>
        \  |  /            /|\
         \ | /            / | \
          \|/
           O =============[===]=============> Output Signal
          /|\
         / | \
        /  |  \
       /   |   \
   
   Receives         Sums signals,        Sends signal to
   signals from     fires if threshold   other neurons
   other neurons    is exceeded
```

Key biological concepts that inspired artificial neural networks:

| Biological | Artificial Equivalent |
|------------|----------------------|
| Dendrites | Input connections |
| Synapse strength | Weights |
| Cell body summation | Weighted sum |
| Firing threshold | Activation function |
| Axon output | Neuron output |

### The Artificial Neuron

An **artificial neuron** (also called a perceptron or node) is a mathematical function that:

1. Receives multiple inputs
2. Multiplies each input by a weight
3. Sums the weighted inputs plus a bias
4. Applies an activation function
5. Produces an output

```
Artificial Neuron:

   x1 ----w1---\
                \
   x2 ----w2------> [SUM + bias] --> [Activation] --> output
                /
   x3 ----w3---/

Mathematical form:
   z = (w1*x1) + (w2*x2) + (w3*x3) + bias
   output = activation(z)
```

**Key Insight**: The weights determine what the neuron "pays attention to." Learning means adjusting these weights to produce correct outputs.

### How Networks Learn Representations

The power of neural networks comes from stacking neurons into layers. Each layer transforms the input, extracting increasingly abstract features:

```
Image Recognition Example:

Input Layer    Hidden Layer 1    Hidden Layer 2    Output Layer
(raw pixels)   (edges)           (shapes)          (object class)

[pixel data] -> [lines, curves] -> [eyes, ears] -> ["cat" vs "dog"]

Each layer builds on the previous layer's abstractions.
```

This is called **representation learning** - the network automatically discovers useful features from raw data, unlike traditional ML where humans engineer features.

```python
# Traditional ML: Human-engineered features
features = extract_edges(image)
features += count_corners(image)
features += measure_symmetry(image)
prediction = classifier.predict(features)

# Neural Network: Learned features
raw_pixels = image.flatten()
prediction = neural_network.predict(raw_pixels)
# Network learns what features matter internally
```

### The Neural Network Zoo

Neural networks come in many architectures, each suited to different problems:

| Architecture | Use Case | Key Feature |
|--------------|----------|-------------|
| Feedforward (MLP) | Tabular data, basic classification | Simple, fully connected |
| Convolutional (CNN) | Images, spatial data | Detects local patterns |
| Recurrent (RNN) | Sequences, time series | Has memory of past inputs |
| Transformer | Text, language | Attention mechanism |
| Autoencoder | Compression, generation | Encodes then decodes |

We'll focus on feedforward networks today and Friday will cover CNNs.

### Historical Context: AI Winters and the Deep Learning Revolution

Understanding the history helps appreciate why neural networks dominate today:

**1943-1969: The Birth and First Hype**
- 1943: McCulloch & Pitts describe first artificial neuron model
- 1958: Rosenblatt invents the Perceptron
- Optimism peaks: "Machines will think within 20 years!"

**1969-1980s: First AI Winter**
- 1969: Minsky & Papert publish "Perceptrons" - prove single perceptrons can't solve XOR
- Funding dries up, research stagnates

**1986-1995: Backpropagation Revival**
- 1986: Rumelhart, Hinton, Williams popularize backpropagation
- Multi-layer networks can solve XOR and more
- Limited by compute power

**1995-2012: Second AI Winter (for neural networks)**
- Support Vector Machines dominate
- Neural networks seen as computationally impractical

**2012-Present: Deep Learning Revolution**
- 2012: AlexNet wins ImageNet by huge margin using GPU-trained deep CNN
- Massive datasets (internet scale)
- GPU computing makes deep networks feasible
- Breakthrough after breakthrough: AlphaGo, GPT, DALL-E, ChatGPT

```
Why Deep Learning Exploded:

Compute Power ----+
                  |
Big Data ---------+---> Deep Learning Success
                  |
Algorithm Advances+
```

### Why "Deep" Learning?

"Deep" refers to networks with many hidden layers:

```
Shallow Network (1-2 hidden layers):
Input --> [Hidden] --> Output

Deep Network (many hidden layers):
Input --> [H1] --> [H2] --> [H3] --> ... --> [Hn] --> Output
```

Deep networks can learn hierarchical representations:
- Early layers: simple patterns (edges, textures)
- Middle layers: combinations (shapes, parts)
- Later layers: complex concepts (objects, scenes)

### Neural Networks vs. Traditional ML

| Aspect | Traditional ML | Neural Networks |
|--------|---------------|-----------------|
| Feature engineering | Manual | Automatic |
| Data requirements | Moderate | Large |
| Interpretability | Often high | Often low ("black box") |
| Compute requirements | Moderate | High |
| Performance ceiling | Limited | State-of-the-art |
| Training time | Fast | Slow |

**When to use each:**
- Neural networks: Large datasets, complex patterns, unstructured data (images, text, audio)
- Traditional ML: Small datasets, tabular data, need for interpretability

## Code Example: A Conceptual Neural Network

```python
import numpy as np

class SimpleNeuron:
    """A single artificial neuron."""
    
    def __init__(self, num_inputs):
        # Initialize random weights and bias
        self.weights = np.random.randn(num_inputs)
        self.bias = np.random.randn()
    
    def forward(self, inputs):
        """Process inputs through the neuron."""
        # Step 1: Weighted sum
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        
        # Step 2: Activation (using sigmoid for now)
        output = 1 / (1 + np.exp(-weighted_sum))
        
        return output

# Create a neuron with 3 inputs
neuron = SimpleNeuron(num_inputs=3)

# Feed it some data
inputs = np.array([0.5, 0.3, 0.2])
output = neuron.forward(inputs)

print(f"Inputs: {inputs}")
print(f"Weights: {neuron.weights}")
print(f"Bias: {neuron.bias:.3f}")
print(f"Output: {output:.3f}")

# The network "learns" by adjusting weights and bias
# to minimize prediction errors (we'll cover this tomorrow!)
```

**Sample Output:**
```
Inputs: [0.5 0.3 0.2]
Weights: [-0.234  0.891  1.234]
Bias: 0.567
Output: 0.723
```

### Multiple Neurons: A Layer

```python
class DenseLayer:
    """A layer of multiple neurons."""
    
    def __init__(self, num_inputs, num_neurons):
        # Each neuron has its own weights + one bias per neuron
        self.weights = np.random.randn(num_inputs, num_neurons) * 0.1
        self.biases = np.zeros(num_neurons)
    
    def forward(self, inputs):
        """Process inputs through all neurons in the layer."""
        # Matrix multiplication handles all neurons at once
        z = np.dot(inputs, self.weights) + self.biases
        # Activation
        return 1 / (1 + np.exp(-z))

# Create a layer with 3 inputs and 4 neurons
layer = DenseLayer(num_inputs=3, num_neurons=4)

inputs = np.array([0.5, 0.3, 0.2])
outputs = layer.forward(inputs)

print(f"Layer input shape: {inputs.shape}")
print(f"Layer output shape: {outputs.shape}")
print(f"Layer outputs: {outputs}")
```

### Stacking Layers: A Network

```python
class SimpleNetwork:
    """A minimal feedforward neural network."""
    
    def __init__(self):
        self.layer1 = DenseLayer(num_inputs=3, num_neurons=4)  # Hidden
        self.layer2 = DenseLayer(num_inputs=4, num_neurons=2)  # Output
    
    def forward(self, inputs):
        """Forward pass through the network."""
        x = self.layer1.forward(inputs)  # Input -> Hidden
        x = self.layer2.forward(x)        # Hidden -> Output
        return x

# Create and use the network
network = SimpleNetwork()
inputs = np.array([0.5, 0.3, 0.2])
outputs = network.forward(inputs)

print(f"Network input: {inputs}")
print(f"Network output: {outputs}")
print(f"Prediction: Class {np.argmax(outputs)}")
```

## The Learning Question

We've seen how data flows through a network (forward propagation), but a crucial question remains: **How do the weights get set correctly?**

Random weights produce random outputs. The magic happens during **training**, where the network:
1. Makes predictions
2. Measures errors
3. Adjusts weights to reduce errors
4. Repeats thousands of times

This process - called **backpropagation** combined with **gradient descent** - is covered in Week 2. For now, understand that learning means systematically adjusting weights to minimize prediction errors.

## Key Takeaways

1. **Neural networks are inspired by biological neurons** but are mathematical functions, not brain simulations.

2. **An artificial neuron computes**: weighted sum of inputs + bias, then applies an activation function.

3. **Networks learn representations** - each layer transforms data into more useful abstractions.

4. **Deep networks have many layers** - enabling hierarchical feature learning from raw data.

5. **History matters** - understanding AI winters and the deep learning revolution contextualizes why neural networks dominate today.

## Looking Ahead

Today's remaining readings dive deeper into the components you've just seen:
- **The Perceptron Model**: Detailed math of a single neuron
- **Activation Functions**: Why non-linearity matters
- **Multi-Layer Perceptrons**: Network architecture
- **Forward Propagation**: How data flows through layers
- **Loss Functions**: Measuring prediction errors

By day's end, you'll understand every component of a neural network - preparing you for Thursday's hands-on TensorFlow implementation.

## Additional Resources

- [3Blue1Brown: Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) - Outstanding visual explanations
- [Neural Network Playground](https://playground.tensorflow.org/) - Interactive visualization
- [A Brief History of Neural Networks](https://cs.stanford.edu/people/eroberts/courses/soco/projects/neural-networks/History/history1.html) - Stanford historical overview

