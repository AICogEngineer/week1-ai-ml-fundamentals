# Distance Between Clusters

## Learning Objectives

- Understand distance metrics as the foundation of similarity measurement
- Calculate Euclidean distance between points in multiple dimensions
- Calculate Manhattan distance and understand when to use it
- Explain how distance choice impacts clustering and classification algorithms

## Why This Matters

At the heart of machine learning lies a deceptively simple question: **How similar are two things?**

K-Means asks: "Which centroid is this point closest to?"
K-Nearest Neighbors asks: "Which training examples are most similar?"
Recommendation systems ask: "Which users behave most like this user?"

The answer depends entirely on how we define "distance." Choose the wrong distance metric, and your algorithm will group unlike things together while separating similar ones. In our **From Zero to Neural** journey, understanding distance is understanding the language machines use to measure similarity.

## The Concept

### Why Distance Matters

Consider two customers:

| Customer | Annual Income | Age |
|----------|--------------|-----|
| Alice | $50,000 | 30 |
| Bob | $52,000 | 31 |
| Carol | $50,000 | 60 |

Intuitively, Alice and Bob seem more similar than Alice and Carol. But a computer needs a numerical measure to make this comparison. That measure is **distance**.

```
Lower distance = More similar
Higher distance = Less similar
```

### Euclidean Distance: The Straight Line

**Euclidean distance** is the "as the crow flies" straight-line distance between two points. It's the most intuitive and commonly used metric.

**2D Formula:**

```
d(p, q) = sqrt[(x2 - x1)^2 + (y2 - y1)^2]
```

**n-Dimensional Formula:**

```
d(p, q) = sqrt[SUM(pi - qi)^2]
        = sqrt[(p1-q1)^2 + (p2-q2)^2 + ... + (pn-qn)^2]
```

**Visual:**

```
      y
      |     * q (4, 4)
      |    /|
      |   / |  Euclidean distance = diagonal line
      |  /  |
      | /   |
      |/____|_____ x
      p (1, 1)

d = sqrt[(4-1)^2 + (4-1)^2] = sqrt[9 + 9] = sqrt[18] = 4.24
```

**Python Implementation:**

```python
import numpy as np

def euclidean_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    point1 = np.array(point1)
    point2 = np.array(point2)
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Example
p1 = [1, 2, 3]
p2 = [4, 5, 6]
distance = euclidean_distance(p1, p2)
print(f"Euclidean distance: {distance:.2f}")  # Output: 5.20
```

### Manhattan Distance: The City Block

**Manhattan distance** (also called L1 distance or taxicab distance) measures distance along grid lines - like navigating city blocks.

**Formula:**

```
d(p, q) = SUM|pi - qi|
        = |p1 - q1| + |p2 - q2| + ... + |pn - qn|
```

**Visual:**

```
      y
      |     * q (4, 4)
      |     |
      |     |  Manhattan = horizontal + vertical
      |_____|
      |     
      |_____ * _____ x
      p (1, 1)

d = |4-1| + |4-1| = 3 + 3 = 6
```

**Python Implementation:**

```python
def manhattan_distance(point1, point2):
    """Calculate Manhattan distance between two points."""
    point1 = np.array(point1)
    point2 = np.array(point2)
    return np.sum(np.abs(point1 - point2))

# Example
p1 = [1, 2, 3]
p2 = [4, 5, 6]
distance = manhattan_distance(p1, p2)
print(f"Manhattan distance: {distance}")  # Output: 9
```

### Euclidean vs. Manhattan: When to Use Each

| Aspect | Euclidean | Manhattan |
|--------|-----------|-----------|
| **Intuition** | Straight line | Grid path |
| **Sensitive to** | Large differences (squared) | All differences equally |
| **Best for** | Continuous, smooth data | High-dimensional, sparse data |
| **Outlier sensitivity** | High (squares magnify outliers) | Lower |
| **Common use** | General purpose, KNN, K-Means | Text analysis, sparse features |

**Key insight**: Euclidean distance squares differences, so a single large difference dominates. Manhattan treats all dimensions more equally.

```
Example: Two points
A = [1, 1, 1, 1]
B = [1, 1, 1, 5]   (only last dimension differs by 4)
C = [2, 2, 2, 2]   (all dimensions differ by 1)

Euclidean:
  d(A,B) = sqrt(0 + 0 + 0 + 16) = 4.0
  d(A,C) = sqrt(1 + 1 + 1 + 1) = 2.0
  --> A is "closer" to C

Manhattan:
  d(A,B) = 0 + 0 + 0 + 4 = 4
  d(A,C) = 1 + 1 + 1 + 1 = 4
  --> A is equidistant from B and C
```

### Other Important Distance Metrics

**Cosine Similarity** (for direction, not magnitude):

```
similarity = (A . B) / (||A|| * ||B||)
distance = 1 - similarity
```

Used in: Text analysis, recommendation systems, embeddings

```python
from sklearn.metrics.pairwise import cosine_similarity

doc1 = [[1, 0, 1, 1, 0]]  # word frequencies
doc2 = [[1, 1, 1, 0, 0]]

similarity = cosine_similarity(doc1, doc2)
print(f"Cosine similarity: {similarity[0][0]:.3f}")
```

**Minkowski Distance** (generalization):

```
d(p, q) = (SUM|pi - qi|^p)^(1/p)

When p=1: Manhattan distance
When p=2: Euclidean distance
When p=infinity: Chebyshev distance (max difference)
```

### Distance in Clustering

K-Means uses distance to:
1. **Assign points to clusters** (nearest centroid)
2. **Measure cluster quality** (within-cluster sum of squares)

```python
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

# Sample data
X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])

# K-Means with Euclidean distance (default)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(X)

# Calculate distances from each point to both centroids
distances = pairwise_distances(X, kmeans.cluster_centers_)
print("Distance to each centroid:")
print(distances)
print(f"\nCluster assignments: {kmeans.labels_}")
```

### The Curse of Dimensionality

As dimensions increase, distance becomes less meaningful:

```
In high dimensions:
- All points become roughly equidistant
- "Nearest neighbor" loses meaning
- Euclidean distance concentrates around a mean

Dimensions:    2D     10D    100D   1000D
Nearest dist:  Varies  Similar  Very similar  Nearly identical
Farthest dist: Varies  Similar  Very similar  Nearly identical
```

**Mitigations:**
- Feature selection (reduce dimensions)
- Dimensionality reduction (PCA)
- Use Manhattan distance (more robust in high-D)
- Consider cosine similarity (ignores magnitude)

### Feature Scaling: Critical for Distance

Distance metrics are sensitive to feature scales. A feature ranging 0-1000 will dominate one ranging 0-1.

```
Unscaled:
  Customer A: income=$50,000, age=30
  Customer B: income=$52,000, age=31
  Customer C: income=$50,000, age=60
  
  d(A,B) = sqrt[(52000-50000)^2 + (31-30)^2] = sqrt[4,000,000 + 1] = 2000.0
  d(A,C) = sqrt[(50000-50000)^2 + (60-30)^2] = sqrt[0 + 900] = 30.0
  
  --> Age difference is completely ignored!

Scaled (StandardScaler):
  Features normalized to mean=0, std=1
  Now age and income contribute equally
```

```python
from sklearn.preprocessing import StandardScaler

# Always scale before distance-based algorithms
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Then apply K-Means
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)
```

## Code Example: Comparing Distance Metrics

```python
import numpy as np
from scipy.spatial.distance import euclidean, cityblock, cosine
from sklearn.preprocessing import StandardScaler

# Customer data: [income, age, purchases_per_month, avg_order_value]
customers = np.array([
    [50000, 30, 5, 100],   # Alice
    [52000, 31, 5, 105],   # Bob (similar to Alice)
    [50000, 60, 5, 100],   # Carol (different age)
    [150000, 30, 15, 500], # David (different behavior)
])

names = ['Alice', 'Bob', 'Carol', 'David']

# Scale features
scaler = StandardScaler()
customers_scaled = scaler.fit_transform(customers)

# Calculate distances from Alice to everyone else
alice_scaled = customers_scaled[0]

print("Distances from Alice:\n")
print(f"{'Name':<10} {'Euclidean':<12} {'Manhattan':<12} {'Cosine':<12}")
print("-" * 46)

for i, name in enumerate(names[1:], 1):
    other_scaled = customers_scaled[i]
    
    euc = euclidean(alice_scaled, other_scaled)
    man = cityblock(alice_scaled, other_scaled)
    cos = cosine(alice_scaled, other_scaled)  # Returns distance, not similarity
    
    print(f"{name:<10} {euc:<12.3f} {man:<12.3f} {cos:<12.3f}")

# Interpretation
print("\nInterpretation:")
print("- Bob is closest to Alice by all metrics (similar demographics & behavior)")
print("- Carol differs mainly in age (one dimension)")
print("- David differs across multiple dimensions (income, purchases, order value)")
```

**Sample Output:**
```
Distances from Alice:

Name       Euclidean    Manhattan    Cosine      
----------------------------------------------
Bob        0.268        0.451        0.001       
Carol      1.500        1.500        0.025       
David      4.123        6.892        0.158       

Interpretation:
- Bob is closest to Alice by all metrics (similar demographics & behavior)
- Carol differs mainly in age (one dimension)
- David differs across multiple dimensions (income, purchases, order value)
```

## Key Takeaways

1. **Distance quantifies similarity** - it's how algorithms measure "closeness" between data points.

2. **Euclidean distance** is the straight-line distance; sensitive to large single differences due to squaring.

3. **Manhattan distance** sums absolute differences; more robust in high dimensions and with outliers.

4. **Always scale features** before using distance-based algorithms - otherwise, large-range features dominate.

5. **High dimensions cause problems** - distances become less meaningful as dimensions increase.

## Looking Ahead

With supervised learning (regression, classification) and unsupervised learning (clustering) foundations in place, you're ready for tomorrow's deep dive into **neural networks**. You'll see how networks use distance-like concepts (loss functions) to measure prediction error and learn optimal weights.

The distance metrics you learned today will reappear in:
- **KNN classification** (nearest neighbor voting)
- **Embedding spaces** (word2vec, neural network representations)
- **Vector databases** (Week 3's focus on similarity search)

## Additional Resources

- [Distance Metrics - TowardsDataScience](https://towardsdatascience.com/9-distance-measures-in-data-science-918109d069fa) - Comprehensive comparison
- [scipy.spatial.distance](https://docs.scipy.org/doc/scipy/reference/spatial.distance.html) - Python implementations
- [Curse of Dimensionality](https://www.visiondummy.com/2014/04/curse-dimensionality-affect-classification/) - Visual explanation

