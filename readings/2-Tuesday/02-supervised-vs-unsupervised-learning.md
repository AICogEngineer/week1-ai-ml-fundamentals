# Supervised vs. Unsupervised Learning

## Learning Objectives

- Distinguish between supervised and unsupervised learning paradigms
- Identify the key characteristics of labeled vs. unlabeled data
- Apply a decision framework for choosing the appropriate approach
- Recognize common algorithms in each category

## Why This Matters

In our **From Zero to Neural** journey, understanding when to use supervised versus unsupervised learning is one of the most critical decisions you'll make as an AI practitioner. Choose wrong, and you'll waste time, resources, and potentially deliver useless results.

This isn't just academic - it's practical. When a business asks you to "find insights in our customer data," you need to know whether they're asking for predictions (supervised) or pattern discovery (unsupervised). The answer determines everything: what data you need, which algorithms to consider, and how you'll measure success.

## The Concept

### The Fundamental Distinction

The difference between supervised and unsupervised learning comes down to one question:

**Do you have labeled data?**

| Aspect | Supervised Learning | Unsupervised Learning |
|--------|--------------------|-----------------------|
| **Data** | Labeled (input + correct output) | Unlabeled (input only) |
| **Goal** | Predict outcomes | Discover patterns |
| **Feedback** | Explicit (we know the "right answer") | None (no ground truth) |
| **Evaluation** | Compare predictions to actual | Subjective / domain expertise |

### Supervised Learning: Learning with a Teacher

In supervised learning, you train a model using examples where you already know the correct answer. The model learns by comparing its predictions to the actual outcomes and adjusting accordingly.

**Analogy**: A student learning with a teacher who grades their work. The student attempts problems, receives feedback on what's right or wrong, and improves.

```
Supervised Learning:

    Training Phase:
    [Features] + [Labels] --> Algorithm --> Model
    
    Prediction Phase:
    [New Features] --> Model --> [Predicted Label]
```

**Example Dataset (Supervised):**

| Square Feet | Bedrooms | Age | Price (Label) |
|-------------|----------|-----|---------------|
| 1500 | 3 | 10 | $250,000 |
| 2000 | 4 | 5 | $350,000 |
| 1200 | 2 | 20 | $180,000 |

The **label** (Price) is what we want to predict. The algorithm learns the relationship between features and labels.

**Two Types of Supervised Learning:**

1. **Regression**: Predict continuous values
   - House prices, temperature, stock prices
   - *Covered in today's reading: "Regression Logic and Applications"*

2. **Classification**: Predict discrete categories
   - Spam/not spam, disease/healthy, cat/dog
   - *Covered in today's reading: "Classification Logic and Applications"*

```python
# Supervised Learning Example: Classification
from sklearn.neighbors import KNeighborsClassifier

# Features (measurements) and labels (species)
X = [[5.1, 3.5], [4.9, 3.0], [7.0, 3.2], [6.4, 3.2]]  # sepal length, width
y = ['setosa', 'setosa', 'versicolor', 'versicolor']   # species labels

# Train the model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# Predict species for a new flower
new_flower = [[5.5, 3.1]]
prediction = model.predict(new_flower)
print(f"Predicted species: {prediction[0]}")
```

### Unsupervised Learning: Learning without a Teacher

In unsupervised learning, you only have input data - no labels, no "correct answers." The algorithm must find patterns, structure, or groupings on its own.

**Analogy**: A student given a box of objects and asked to organize them into groups. No teacher tells them the "right" way - they must discover meaningful categories themselves.

```
Unsupervised Learning:

    Training Phase:
    [Features Only] --> Algorithm --> Discovered Patterns
    
    Application Phase:
    [New Data] --> Patterns --> [Group Assignment / Insight]
```

**Example Dataset (Unsupervised):**

| Customer ID | Annual Spend | Visit Frequency | Avg Purchase |
|-------------|--------------|-----------------|--------------|
| 001 | $5,000 | 50 | $100 |
| 002 | $500 | 5 | $100 |
| 003 | $4,800 | 48 | $100 |

Notice: No labels! We don't know which customers are "valuable" or "at-risk." The algorithm discovers natural groupings.

**Common Unsupervised Tasks:**

1. **Clustering**: Group similar data points
   - Customer segmentation, document grouping
   - *Covered in today's reading: "K-Means Clustering"*

2. **Dimensionality Reduction**: Compress features while preserving information
   - PCA for visualization, feature reduction

3. **Anomaly Detection**: Find unusual data points
   - Fraud detection, system monitoring

```python
# Unsupervised Learning Example: Clustering
from sklearn.cluster import KMeans
import numpy as np

# Customer data: [annual_spend, visit_frequency]
customers = np.array([
    [5000, 50], [4800, 48], [5200, 52],  # High-value cluster
    [500, 5], [600, 6], [450, 4],        # Low-value cluster
    [2500, 25], [2600, 26], [2400, 24]   # Medium-value cluster
])

# Find 3 clusters
model = KMeans(n_clusters=3, random_state=42)
model.fit(customers)

# See which cluster each customer belongs to
print(f"Cluster assignments: {model.labels_}")
print(f"Cluster centers: {model.cluster_centers_}")
```

### Side-by-Side Comparison

```
SUPERVISED LEARNING                 UNSUPERVISED LEARNING
==================                  =====================

Input: Features + Labels            Input: Features Only
       [X, y]                              [X]

        |                                   |
        v                                   v

Algorithm learns mapping            Algorithm finds structure
from X to y                         within X

        |                                   |
        v                                   v

Output: Predictions                 Output: Patterns/Groups
        y_new = f(X_new)                   clusters, reduced dims

        |                                   |
        v                                   v

Evaluation: Compare to              Evaluation: Domain expertise
actual labels                       or internal metrics
```

### Decision Framework: Which Approach to Use?

Use this flowchart when approaching a new problem:

```
START
  |
  v
Do you have labeled data?
  |
  +-- YES --> Do you want to predict categories or values?
  |             |
  |             +-- Categories --> CLASSIFICATION
  |             |                  (Supervised)
  |             |
  |             +-- Values --> REGRESSION
  |                            (Supervised)
  |
  +-- NO --> What do you want to achieve?
              |
              +-- Group similar items --> CLUSTERING
              |                           (Unsupervised)
              |
              +-- Reduce features --> DIMENSIONALITY REDUCTION
              |                       (Unsupervised)
              |
              +-- Find outliers --> ANOMALY DETECTION
                                    (Unsupervised)
```

### Real-World Scenarios

| Scenario | Data Available | Approach | Why |
|----------|---------------|----------|-----|
| Predict customer churn | Historical data with churn labels | Supervised (Classification) | We know who churned |
| Segment customers for marketing | Customer behavior data, no segments defined | Unsupervised (Clustering) | No predefined groups |
| Predict house prices | Historical sales with prices | Supervised (Regression) | We have price labels |
| Detect fraudulent transactions | Mostly normal transactions, few labeled fraud | Unsupervised (Anomaly Detection) | Too few labeled examples |
| Recommend products | User-item interaction data | Can be either | Depends on approach |

### Combining Approaches: Semi-Supervised Learning

In practice, you often have some labeled data and lots of unlabeled data. Semi-supervised learning leverages both:

1. Use unsupervised learning to understand data structure
2. Use limited labels to guide learning
3. Propagate labels to similar unlabeled points

This is increasingly common in industry where labeling is expensive.

## Key Takeaways

1. **Supervised learning requires labels** - you need known outcomes to train the model to predict new outcomes.

2. **Unsupervised learning discovers structure** - no labels needed, but evaluation is more subjective.

3. **The choice depends on your data and goal**:
   - Have labels + want predictions = Supervised
   - No labels + want patterns = Unsupervised

4. **Supervised learning is easier to evaluate** - you can measure accuracy against known answers.

5. **Unsupervised learning requires domain expertise** - interpreting discovered patterns needs human judgment.

## Looking Ahead

Today's remaining readings will dive deep into specific algorithms:
- **Regression and Classification** expand on supervised learning
- **K-Means Clustering** introduces unsupervised learning in practice

By Wednesday, you'll see how neural networks can be applied to both supervised tasks (classification, regression) and even unsupervised tasks (autoencoders - which we'll touch on in Week 2).

## Additional Resources

- [Supervised vs Unsupervised Learning - IBM](https://www.ibm.com/think/topics/supervised-vs-unsupervised-learning) - Clear industry perspective
- [scikit-learn Algorithm Cheat Sheet](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html) - Visual guide for algorithm selection
- [Google ML Crash Course: Framing](https://developers.google.com/machine-learning/crash-course/framing/video-lecture) - How to frame ML problems

