# Introduction to Machine Learning Algorithms

## Learning Objectives

- Define machine learning and articulate how it differs from traditional programming
- Understand the fundamental learning paradigm: data + algorithm = model
- Identify the three main categories of machine learning algorithms
- Recognize when machine learning is (and isn't) the right tool for a problem

## Why This Matters

In our journey **From Zero to Neural**, understanding what machine learning actually *is* forms the bedrock of everything that follows. Before we can build neural networks or train models on GPUs, we need to fundamentally shift how we think about solving problems with computers.

Traditional programming has served us well for decades: you analyze a problem, write explicit rules, and the computer follows those rules precisely. But what happens when the rules are too complex to write? How do you program a computer to recognize a cat in a photo, detect fraudulent transactions, or predict tomorrow's weather?

Machine learning flips the paradigm. Instead of programming rules, we provide examples and let the algorithm discover the rules itself.

## The Concept

### What Is Machine Learning?

**Machine Learning (ML)** is a subset of artificial intelligence where computers learn patterns from data without being explicitly programmed for every scenario.

The formal definition from Tom Mitchell (1997):

> "A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E."

Let's break this down with a concrete example:

| Component | Spam Detection Example |
|-----------|----------------------|
| **Task (T)** | Classify emails as spam or not spam |
| **Experience (E)** | A dataset of emails labeled as spam/not spam |
| **Performance (P)** | Percentage of emails correctly classified |

The system "learns" by analyzing thousands of labeled emails, discovering patterns (certain words, sender behaviors, formatting), and using those patterns to classify new, unseen emails.

### Traditional Programming vs. Machine Learning

```
Traditional Programming:
    Input Data + Rules  -->  [Computer]  -->  Output

Machine Learning:
    Input Data + Output -->  [Computer]  -->  Rules (Model)
```

**Traditional Approach (Spam Filter):**
```python
def is_spam(email):
    spam_words = ["free", "winner", "click here", "limited time"]
    for word in spam_words:
        if word in email.lower():
            return True
    return False
```

The problem? Spammers adapt. They use "fr3e" or "w1nner". You're constantly updating rules.

**Machine Learning Approach:**
```python
from sklearn.naive_bayes import MultinomialNB

# The algorithm learns patterns from examples
model = MultinomialNB()
model.fit(training_emails, training_labels)  # Learn from data

# Now it can classify new emails based on learned patterns
prediction = model.predict(new_email)
```

The model discovers patterns you might never have explicitly programmed.

### The Learning Paradigm: Data + Algorithm = Model

Every machine learning system follows this fundamental equation:

```
Training Data + Learning Algorithm = Trained Model
```

1. **Training Data**: Historical examples with known outcomes
2. **Learning Algorithm**: The mathematical procedure that finds patterns
3. **Trained Model**: The resulting "learned" representation that can make predictions

Think of it like this:
- **Data** is the textbook
- **Algorithm** is the student's learning method
- **Model** is the knowledge the student retains after studying

### Categories of Machine Learning

Machine learning algorithms fall into three main categories:

#### 1. Supervised Learning
- **What it is**: Learning from labeled examples
- **The data**: Input-output pairs (features and target)
- **The goal**: Predict outputs for new inputs
- **Examples**: Predicting house prices, classifying emails, diagnosing diseases

We'll explore supervised learning in depth throughout today's readings on regression and classification.

#### 2. Unsupervised Learning
- **What it is**: Finding patterns in unlabeled data
- **The data**: Inputs only (no target labels)
- **The goal**: Discover hidden structure
- **Examples**: Customer segmentation, anomaly detection, topic modeling

Today's reading on K-Means clustering will introduce this category.

#### 3. Reinforcement Learning
- **What it is**: Learning through trial and error with rewards
- **The data**: Environment feedback (rewards/penalties)
- **The goal**: Maximize cumulative reward
- **Examples**: Game-playing AI, robotics, autonomous vehicles

*Note: Reinforcement learning is beyond the scope of this week's curriculum.*

### When to Use Machine Learning

Machine learning shines when:

| Use ML When... | Example |
|----------------|---------|
| Rules are too complex to code | Facial recognition |
| Rules change frequently | Fraud detection (fraudsters adapt) |
| The problem is personalization | Recommendation systems |
| You have lots of data | Predictive maintenance with sensor data |
| Humans can do it but can't explain how | Speech recognition |

Avoid ML when:
- Simple rules suffice
- You have very little data
- Explainability is legally required and the task is simple
- The cost of errors is catastrophic and the model isn't proven

## Code Example: The ML Workflow

Here's a minimal example showing the complete ML workflow:

```python
# Step 1: Import the algorithm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

# Step 2: Prepare data (square footage -> price)
square_feet = np.array([1000, 1500, 2000, 2500, 3000]).reshape(-1, 1)
prices = np.array([150000, 200000, 250000, 300000, 350000])

# Step 3: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    square_feet, prices, test_size=0.2, random_state=42
)

# Step 4: Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)  # This is where learning happens

# Step 5: Make predictions
new_house = np.array([[1800]])
predicted_price = model.predict(new_house)
print(f"Predicted price for 1800 sq ft: ${predicted_price[0]:,.2f}")

# Step 6: Evaluate the model
score = model.score(X_test, y_test)
print(f"Model accuracy (R-squared): {score:.2f}")
```

Output:
```
Predicted price for 1800 sq ft: $230,000.00
Model accuracy (R-squared): 1.00
```

## Key Takeaways

1. **Machine learning is pattern recognition at scale** - algorithms discover rules from data rather than following hand-coded rules.

2. **The paradigm shift**: Traditional programming encodes human knowledge as rules; ML extracts knowledge from data.

3. **Three ingredients**: Training data + learning algorithm = trained model.

4. **Three categories**: Supervised (labeled data), unsupervised (unlabeled data), and reinforcement learning (reward-based).

5. **ML is a tool, not magic** - it requires quality data, appropriate algorithm selection, and careful evaluation.

## Looking Ahead

With this foundation in place, you're ready to explore the two pillars of supervised learning: **regression** (predicting continuous values) and **classification** (predicting categories). These concepts form the building blocks for everything from simple linear models to the deep neural networks you'll build by Friday.

## Additional Resources

- [Google's Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course) - Comprehensive free course with interactive exercises
- [scikit-learn Documentation: Getting Started](https://scikit-learn.org/stable/getting_started.html) - Practical introduction to ML in Python
- [A Visual Introduction to Machine Learning](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/) - Beautiful visual explanation of ML concepts

