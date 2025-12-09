"""
Exercise 08: Sequential Model Builder
=====================================

PAIR PROGRAMMING ACTIVITY

Partner A: _______________
Partner B: _______________

Remember to switch Driver/Navigator roles every 15-20 minutes!
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("=" * 60)
print("SEQUENTIAL MODEL BUILDER - PAIR PROGRAMMING")
print("=" * 60)

# =============================================================================
# MODEL 1: Simple Binary Classifier
# =============================================================================
# Driver: _______________
# Navigator: _______________

print("\n--- MODEL 1: Binary Classifier ---")

# Requirements:
# - Input: 20 features
# - Output: Binary classification
# - At least 2 hidden layers
# - ReLU in hidden layers

model_1 = keras.Sequential([
    # TODO: Add layers
], name='binary_classifier')

# model_1.compile(
#     optimizer='adam',
#     loss='binary_crossentropy',
#     metrics=['accuracy']
# )
# model_1.summary()


# =============================================================================
# MODEL 2: Multi-Class Classifier
# =============================================================================
# Driver: _______________  (SWITCH!)
# Navigator: _______________

print("\n--- MODEL 2: Multi-Class Classifier ---")

# Requirements:
# - Input: 784 features
# - Output: 10 classes
# - Use dropout
# - Parameters: 100,000 - 200,000

model_2 = keras.Sequential([
    # TODO: Add layers
], name='multiclass_classifier')

# model_2.compile(
#     optimizer='adam',
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy']
# )
# model_2.summary()
# assert 100_000 <= model_2.count_params() <= 200_000


# =============================================================================
# MODEL 3: Regression Network
# =============================================================================
# Driver: _______________  (SWITCH!)
# Navigator: _______________

print("\n--- MODEL 3: Regression Network ---")

# Requirements:
# - Input: 13 features
# - Output: Single continuous value
# - Use batch normalization
# - At least 3 hidden layers

model_3 = keras.Sequential([
    # TODO: Add layers
], name='regression_network')

# model_3.compile(
#     optimizer='adam',
#     loss='mse',
#     metrics=['mae']
# )
# model_3.summary()


# =============================================================================
# MODEL 4: Deep Network
# =============================================================================
# Driver: _______________  (SWITCH!)
# Navigator: _______________

print("\n--- MODEL 4: Deep Network ---")

# Requirements:
# - Input: 100 features
# - Output: 5 classes
# - At least 5 hidden layers
# - Use both Dropout AND BatchNormalization
# - Parameters: 50,000 - 100,000

model_4 = keras.Sequential([
    # TODO: Add layers
], name='deep_network')

# model_4.compile(...)
# model_4.summary()
# assert 50_000 <= model_4.count_params() <= 100_000


# =============================================================================
# MODEL 5: Minimal Network Challenge
# =============================================================================
# Driver: _______________  (SWITCH!)
# Navigator: _______________

print("\n--- MODEL 5: Minimal Network ---")

# Requirements:
# - Input: 50 features
# - Output: 3 classes
# - MAXIMUM 1,000 parameters
# - At least 1 hidden layer

model_5 = keras.Sequential([
    # TODO: Add layers
], name='minimal_network')

# model_5.compile(...)
# model_5.summary()
# assert model_5.count_params() <= 1000


# =============================================================================
# COMPARISON TABLE
# =============================================================================

print("\n" + "=" * 60)
print("COMPARISON TABLE")
print("=" * 60)

# Fill this out together:
comparison = """
| Model | Input | Output | Hidden Layers | Parameters | Special Features |
|-------|-------|--------|---------------|------------|------------------|
| 1     |       |        |               |            |                  |
| 2     |       |        |               |            |                  |
| 3     |       |        |               |            |                  |
| 4     |       |        |               |            |                  |
| 5     |       |        |               |            |                  |
"""
print(comparison)


# =============================================================================
# PARTNER FEEDBACK
# =============================================================================

print("\n" + "=" * 60)
print("PARTNER FEEDBACK")
print("=" * 60)

# Partner A Feedback:
# 1. What did you learn from your partner?
#    Answer:
#
# 2. What was the most challenging architecture?
#    Answer:
#
# 3. Did Driver or Navigator feel more natural?
#    Answer:

# Partner B Feedback:
# 1. What did you learn from your partner?
#    Answer:
#
# 2. What was the most challenging architecture?
#    Answer:
#
# 3. Did Driver or Navigator feel more natural?
#    Answer:

