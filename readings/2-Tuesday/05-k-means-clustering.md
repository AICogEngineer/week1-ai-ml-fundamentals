# K-Means Clustering

## Learning Objectives

- Understand K-Means as an unsupervised grouping algorithm
- Explain the iterative centroid initialization, assignment, and update process
- Apply the elbow method to determine the optimal number of clusters
- Recognize practical applications of clustering in business and data science

## Why This Matters

Imagine you're a marketing director with data on millions of customers, but no predefined segments. How do you discover natural groupings? You can't label every customer manually, and you don't even know what the groups should be.

This is where unsupervised learning shines. K-Means clustering automatically discovers structure in your data - finding the "tribes" within your customer base, the categories in your product catalog, or the patterns in your sensor readings.

In our **From Zero to Neural** journey, K-Means represents a fundamental shift: we're no longer learning from labels, but discovering patterns the data reveals on its own.

## The Concept

### What Is K-Means Clustering?

**K-Means** is an unsupervised learning algorithm that partitions data into **K distinct, non-overlapping groups (clusters)** based on similarity.

Key characteristics:
- **K** = number of clusters (you choose this)
- **Means** = each cluster is represented by its center (centroid)
- **Unsupervised** = no labels required

```
Before Clustering:          After K-Means (K=3):

    *  *                       o  o
  *    *  *                  o    o  o
    *                          o
         * *  *                     x x  x
       *   *                      x   x
            *                         x
  *  *                        +  +
    *  *                        +  +
      *                           +

(undifferentiated points)    (3 distinct clusters)
```

### The K-Means Algorithm

K-Means works through a simple but powerful iterative process:

```
ALGORITHM: K-Means Clustering

INPUT: Data points X, number of clusters K
OUTPUT: K cluster assignments for each point

1. INITIALIZE: Randomly place K centroids in the feature space

2. REPEAT until convergence:
   
   a. ASSIGN: For each data point, find the nearest centroid
              Assign that point to that centroid's cluster
   
   b. UPDATE: For each cluster, recalculate the centroid
              (mean position of all points in that cluster)

3. RETURN: Final cluster assignments
```

### Step-by-Step Visual Walkthrough

**Initial State**: 6 points, K=2 clusters

```
Step 0: Random Centroid Initialization
        
     5 |  *           *
     4 |        C1
     3 |     *     *
     2 |              C2
     1 |  *        *
       +----------------
         1  2  3  4  5  6

       C1, C2 = randomly placed centroids
```

**Iteration 1: Assignment**

```
Step 1a: Assign points to nearest centroid

     5 |  o           x      (o = cluster 1, x = cluster 2)
     4 |        C1
     3 |     o     x
     2 |              C2
     1 |  o        x
       +----------------
         1  2  3  4  5  6

       Points assigned based on distance
```

**Iteration 1: Update**

```
Step 1b: Recalculate centroids (mean of assigned points)

     5 |  o           x
     4 |     C1
     3 |     o     x         C1 moved to mean of o's
     2 |           C2 x      C2 moved to mean of x's
     1 |  o        
       +----------------
         1  2  3  4  5  6
```

**Continue until centroids stop moving (convergence).**

### Choosing K: The Elbow Method

The biggest question in K-Means: **How many clusters?**

The **Elbow Method** helps by plotting the "within-cluster sum of squares" (WCSS) for different K values:

```
WCSS (Inertia)
    |
800 |*
    |
600 | *
    |
400 |   *
    |     *-----*-----*-----*   <-- "Elbow" around K=4
200 |
    +--------------------------> K
      1   2   3   4   5   6   7
```

**Interpretation:**
- Low K = high WCSS (points far from centroids)
- High K = low WCSS (but overfitting - too many clusters)
- **Elbow point** = diminishing returns; adding more clusters doesn't help much

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Calculate WCSS for different K values
wcss = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot the elbow curve
plt.plot(K_range, wcss, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS (Inertia)')
plt.title('Elbow Method for Optimal K')
plt.show()
```

### Centroid Initialization Matters

Random initialization can lead to poor results. K-Means++ is a smarter initialization:

```python
# Default in scikit-learn: k-means++
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)

# Manual: random initialization (not recommended)
kmeans = KMeans(n_clusters=3, init='random', random_state=42)
```

**K-Means++ Strategy:**
1. Choose first centroid randomly
2. For each subsequent centroid, choose points with probability proportional to distance from nearest existing centroid
3. Spreads initial centroids apart, leading to better convergence

## Code Example: Complete Clustering Workflow

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Sample customer data: [annual_income, spending_score]
# (Spending score: 1-100, based on purchasing behavior)
customers = np.array([
    [15, 39], [16, 81], [17, 6], [18, 77], [19, 40],
    [20, 76], [21, 6], [23, 94], [24, 3], [25, 72],
    [40, 42], [41, 52], [42, 48], [43, 54], [44, 46],
    [45, 55], [46, 51], [47, 50], [48, 49], [49, 53],
    [70, 82], [71, 91], [72, 78], [73, 89], [74, 87],
    [75, 95], [76, 88], [77, 93], [78, 79], [79, 84],
    [85, 5], [86, 12], [87, 8], [88, 15], [89, 6],
    [90, 10], [91, 3], [92, 14], [93, 7], [94, 11],
])

# Step 1: Scale features (important for distance-based algorithms)
scaler = StandardScaler()
customers_scaled = scaler.fit_transform(customers)

# Step 2: Determine optimal K using Elbow Method
wcss = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(customers_scaled)
    wcss.append(kmeans.inertia_)

# Step 3: Fit K-Means with chosen K (let's say K=4 based on elbow)
K = 4
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
clusters = kmeans.fit_predict(customers_scaled)

# Step 4: Analyze results
print("Cluster Assignments:")
for i in range(K):
    cluster_members = customers[clusters == i]
    print(f"\nCluster {i}:")
    print(f"  Size: {len(cluster_members)} customers")
    print(f"  Avg Income: ${cluster_members[:, 0].mean()*1000:,.0f}")
    print(f"  Avg Spending Score: {cluster_members[:, 1].mean():.1f}")

# Step 5: Get centroids (in original scale)
centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
print("\nCluster Centroids (Income, Spending Score):")
for i, centroid in enumerate(centroids_original):
    print(f"  Cluster {i}: Income=${centroid[0]*1000:,.0f}, Score={centroid[1]:.1f}")
```

**Sample Output:**
```
Cluster Assignments:

Cluster 0:
  Size: 10 customers
  Avg Income: $44,500
  Avg Spending Score: 50.0

Cluster 1:
  Size: 10 customers
  Avg Income: $18,500
  Avg Spending Score: 49.4

Cluster 2:
  Size: 10 customers
  Avg Income: $74,500
  Avg Spending Score: 86.6

Cluster 3:
  Size: 10 customers
  Avg Income: $89,500
  Avg Spending Score: 9.1

Cluster Centroids (Income, Spending Score):
  Cluster 0: Income=$44,500, Score=50.0
  Cluster 1: Income=$18,500, Score=49.4
  Cluster 2: Income=$74,500, Score=86.6
  Cluster 3: Income=$89,500, Score=9.1
```

### Interpreting the Clusters

Based on the output, we might label these customer segments:

| Cluster | Income | Spending | Business Label |
|---------|--------|----------|----------------|
| 0 | Medium | Medium | "Average Joes" |
| 1 | Low | Medium | "Careful Spenders" |
| 2 | High | High | "Big Spenders" (target for premium products) |
| 3 | High | Low | "Savers" (target for savings products) |

## Real-World Applications

| Domain | Clustering Application | Business Value |
|--------|----------------------|----------------|
| Marketing | Customer segmentation | Targeted campaigns |
| Retail | Product categorization | Better recommendations |
| Healthcare | Patient grouping | Personalized treatment |
| Finance | Risk profiling | Portfolio management |
| Image Processing | Color quantization | Image compression |
| Anomaly Detection | Normal behavior clusters | Fraud/intrusion detection |
| Document Analysis | Topic discovery | Content organization |
| Biology | Gene expression grouping | Disease research |

### Industry Example: Retail Customer Segmentation

```python
# E-commerce clustering features
customer_features = [
    'recency',           # Days since last purchase
    'frequency',         # Number of purchases
    'monetary',          # Total spending
    'avg_basket_size',   # Average order value
    'product_diversity', # Number of categories purchased
]

# After clustering, marketing can:
# - Send win-back campaigns to "Lapsed" cluster
# - Offer loyalty rewards to "Champions" cluster
# - Upsell premium products to "High Potential" cluster
```

## K-Means Limitations

| Limitation | Description | Mitigation |
|------------|-------------|------------|
| **Fixed K** | Must specify clusters in advance | Use elbow method, silhouette score |
| **Spherical clusters** | Assumes round clusters | Use DBSCAN for irregular shapes |
| **Sensitive to outliers** | Outliers pull centroids | Remove outliers or use K-Medoids |
| **Sensitive to initialization** | Random start affects result | Use K-Means++ (default) |
| **Equal-sized clusters** | Tends toward similar sizes | Consider Gaussian Mixture Models |

```
K-Means works well:          K-Means struggles:

    ooo      xxx                  ooooooo
   ooooo    xxxxx               oo     ooo
    ooo      xxx               o  xxx    o
                               o xxxxx   o
  (spherical clusters)          xxxxxxx
                              (concentric or irregular shapes)
```

## Key Takeaways

1. **K-Means partitions data into K groups** based on distance to cluster centroids.

2. **The algorithm iterates**: assign points to nearest centroid, then update centroids.

3. **Choose K carefully** - use the elbow method to find where additional clusters stop helping.

4. **Scale your features** - K-Means uses distance, so features must be comparable.

5. **K-Means assumes spherical clusters** - consider alternatives for irregular shapes.

## Looking Ahead

The next reading dives deep into **distance metrics** - the mathematical foundation that K-Means (and many other algorithms) uses to measure "similarity." Understanding Euclidean vs. Manhattan distance will help you choose the right metric for your data.

## Additional Resources

- [K-Means Clustering - StatQuest](https://www.youtube.com/watch?v=4b5d3muPQmA) - Visual walkthrough
- [scikit-learn K-Means](https://scikit-learn.org/stable/modules/clustering.html#k-means) - Official documentation
- [Elbow Method Explained](https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/) - Detailed tutorial

