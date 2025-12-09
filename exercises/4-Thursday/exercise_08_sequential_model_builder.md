# Exercise 08: Sequential Model Builder

## PAIR PROGRAMMING ACTIVITY

**This exercise is designed for two people working together!**

---

## Learning Objectives

- Collaborate effectively using Driver/Navigator pattern
- Build diverse Sequential model architectures
- Understand how architecture choices affect model properties
- Practice explaining technical decisions verbally

## Duration

**Estimated Time:** 90 minutes

---

## Pair Programming Roles

### Driver
- Types the code
- Focuses on syntax and implementation
- Asks clarifying questions

### Navigator
- Designs the architecture
- Thinks about the big picture
- Catches errors and suggests improvements

**Switch roles every 15-20 minutes!**

---

## Setup

Both partners should have:
- Shared screen or side-by-side computers
- Same codebase open
- TensorFlow installed

---

## Part 1: Warmup - Build Together (20 min)

### Model 1: Simple Binary Classifier

**Navigator designs, Driver implements:**

Requirements:
- Input: 20 features
- Output: Binary classification (0 or 1)
- At least 2 hidden layers
- Use ReLU in hidden layers

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# DRIVER: Type this model
model_1 = keras.Sequential([
    # NAVIGATOR: Specify the layers
    # layers.Dense(???, activation=???, input_shape=(???))
    # ...
])

model_1.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model_1.summary()
```

### Discussion Points (Both partners)

1. Why did we choose this number of neurons?
2. Why binary_crossentropy loss?
3. What activation is on the output layer?

---

## Part 2: Architecture Challenges (50 min)

**Switch roles for each model!**

### Model 2: Multi-Class Classifier

**Switch roles now!**

Requirements:
- Input: 784 features (flattened 28x28 image)
- Output: 10 classes
- Use dropout for regularization
- Total parameters should be between 100,000 and 200,000

```python
# DRIVER: Implement the architecture
# NAVIGATOR: Guide the design and calculate parameters

model_2 = keras.Sequential([
    # Your design here
])

model_2.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model_2.summary()

# NAVIGATOR: Verify parameter count is in range
total_params = model_2.count_params()
print(f"Total parameters: {total_params:,}")
assert 100_000 <= total_params <= 200_000, "Parameters out of range!"
```

### Model 3: Regression Network

**Switch roles now!**

Requirements:
- Input: 13 features (Boston housing style)
- Output: Single continuous value (price)
- Use batch normalization
- Include at least 3 hidden layers

```python
model_3 = keras.Sequential([
    # Your design here
    # Hint: layers.BatchNormalization() goes after Dense layers
])

model_3.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

model_3.summary()
```

Discussion:
- Why no activation on the output layer?
- Where did you place BatchNormalization?

### Model 4: Deep Network

**Switch roles now!**

Requirements:
- Input: 100 features
- Output: 5 classes
- Must have at least 5 hidden layers
- Must use both Dropout AND BatchNormalization
- Parameters between 50,000 and 100,000

```python
model_4 = keras.Sequential([
    # Your design here
    # Challenge: Balance depth with parameter count
])

model_4.compile(...)
model_4.summary()
```

Discussion:
- How did you keep parameter count low with 5 layers?
- What order: Dense -> BatchNorm -> Dropout or Dense -> Dropout -> BatchNorm?

### Model 5: Minimal Network Challenge

**Switch roles now!**

Requirements:
- Input: 50 features
- Output: 3 classes
- **Maximum 1,000 parameters**
- Must still have at least 1 hidden layer

```python
model_5 = keras.Sequential([
    # Can you fit under 1000 params?
])

model_5.compile(...)
model_5.summary()

assert model_5.count_params() <= 1000, "Too many parameters!"
```

Discussion:
- What trade-offs did you make?
- Would this model likely perform well?

---

## Part 3: Reflection and Documentation (20 min)

### Joint Summary

Together, fill out this comparison table:

```
| Model | Input | Output | Hidden Layers | Parameters | Special Features |
|-------|-------|--------|---------------|------------|------------------|
| 1     |       |        |               |            |                  |
| 2     |       |        |               |            |                  |
| 3     |       |        |               |            |                  |
| 4     |       |        |               |            |                  |
| 5     |       |        |               |            |                  |
```

### Partner Feedback

Each partner answers:

1. **What did you learn from your partner?**
   - Partner A:
   - Partner B:

2. **What was the most challenging architecture to design?**
   - Partner A:
   - Partner B:

3. **Did Driver or Navigator feel more natural to you?**
   - Partner A:
   - Partner B:

---

## Definition of Done

- [ ] All 5 models built and compiled
- [ ] Parameter constraints verified
- [ ] Roles switched at least 4 times
- [ ] Comparison table completed
- [ ] Partner feedback documented

---

## Pair Programming Tips

1. **Communicate constantly** - "I'm thinking of adding 128 neurons..."
2. **No silent coding** - Navigator should always be guiding
3. **Be patient** - Learning together takes time
4. **Celebrate wins** - "Nice, we hit the parameter target!"
5. **Ask questions** - "Why did you choose ReLU there?"

---

## Submission

Both partners submit:
1. Python file with all 5 models
2. Completed comparison table
3. Partner feedback section

Include both names in the submission!

