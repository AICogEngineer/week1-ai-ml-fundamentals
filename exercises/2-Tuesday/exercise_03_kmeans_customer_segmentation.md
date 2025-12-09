# Exercise 03: K-Means Customer Segmentation

## Learning Objectives

- Apply K-Means clustering to a real business problem
- Use the Elbow Method to determine optimal K
- Interpret and name discovered segments
- Create actionable business recommendations

## Duration

**Estimated Time:** 90 minutes

## Background

You're a data scientist at an e-commerce company. Marketing wants to understand customer segments to tailor campaigns. You have data on customer purchasing behavior but NO predefined categories - this is where unsupervised learning shines!

---

## The Scenario

**RapidMart** has provided you with anonymized customer data:

| Feature | Description |
|---------|-------------|
| `annual_income` | Customer's yearly income (thousands $) |
| `spending_score` | Score 1-100 based on purchase frequency and amount |
| `age` | Customer age |
| `membership_years` | Years as a member |

Your mission: Discover natural customer segments and recommend targeted strategies.

---

## Part 1: Data Exploration (20 min)

### Task 1.1: Load and Examine Data

Navigate to `starter_code/customer_segmentation.py`:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Generate realistic customer data
np.random.seed(42)
n_customers = 500

# Create distinct customer segments (hidden from trainees)
data = pd.DataFrame({
    'customer_id': range(1, n_customers + 1),
    'annual_income': np.concatenate([
        np.random.normal(30, 8, 100),    # Low income
        np.random.normal(55, 10, 150),   # Middle income
        np.random.normal(90, 12, 150),   # High income
        np.random.normal(120, 15, 100)   # Very high income
    ]),
    'spending_score': np.concatenate([
        np.random.normal(25, 10, 100),   # Low spenders
        np.random.normal(50, 15, 150),   # Medium spenders
        np.random.normal(75, 12, 150),   # High spenders
        np.random.normal(45, 20, 100)    # Variable (wealthy but selective)
    ]),
    'age': np.concatenate([
        np.random.normal(22, 3, 100),    # Young
        np.random.normal(35, 8, 150),    # Middle-aged
        np.random.normal(45, 10, 150),   # Mature
        np.random.normal(55, 8, 100)     # Senior
    ]),
    'membership_years': np.concatenate([
        np.random.normal(1, 0.5, 100),
        np.random.normal(3, 1.5, 150),
        np.random.normal(5, 2, 150),
        np.random.normal(8, 2, 100)
    ])
})

# Clip to realistic ranges
data['annual_income'] = data['annual_income'].clip(15, 150)
data['spending_score'] = data['spending_score'].clip(1, 100)
data['age'] = data['age'].clip(18, 70).astype(int)
data['membership_years'] = data['membership_years'].clip(0, 15).round(1)

# TODO: Explore the data
# print(data.head())
# print(data.describe())
```

### Task 1.2: Visualize Relationships

```python
# TODO: Create scatter plots to explore potential clusters
# Focus on: income vs spending, age vs spending, etc.

# plt.figure(figsize=(15, 5))
# plt.subplot(1, 3, 1)
# plt.scatter(data['annual_income'], data['spending_score'], alpha=0.5)
# ...
```

---

## Part 2: Finding Optimal K (25 min)

### Task 2.1: Prepare Data for Clustering

```python
# Select features for clustering
features = ['annual_income', 'spending_score', 'age', 'membership_years']
X = data[features].values

# TODO: Scale the features (CRITICAL for K-Means!)
scaler = StandardScaler()
X_scaled = None  # Your code
```

### Task 2.2: Elbow Method

```python
# TODO: Calculate inertia for K from 1 to 10
K_range = range(1, 11)
inertias = []

for k in K_range:
    # kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    # kmeans.fit(X_scaled)
    # inertias.append(kmeans.inertia_)
    pass

# TODO: Plot the elbow curve
# plt.figure(figsize=(10, 6))
# plt.plot(K_range, inertias, 'bo-')
# plt.xlabel('Number of Clusters (K)')
# plt.ylabel('Inertia')
# plt.title('Elbow Method')
# plt.show()
```

### Task 2.3: Determine Optimal K

```python
# TODO: Based on the elbow plot, what is the optimal K?
optimal_k = None  # Your answer

# Justify your choice in comments:
# Why did you choose this K?
```

---

## Part 3: Clustering and Interpretation (30 min)

### Task 3.1: Apply K-Means

```python
# TODO: Fit K-Means with your chosen K
# kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
# data['cluster'] = kmeans.fit_predict(X_scaled)
```

### Task 3.2: Analyze Clusters

```python
# TODO: Calculate mean values for each cluster
# cluster_summary = data.groupby('cluster')[features].mean()
# print(cluster_summary)

# TODO: Count customers per cluster
# print(data['cluster'].value_counts().sort_index())
```

### Task 3.3: Visualize Clusters

```python
# TODO: Create scatter plot colored by cluster
# plt.figure(figsize=(12, 8))
# for cluster in range(optimal_k):
#     mask = data['cluster'] == cluster
#     plt.scatter(
#         data[mask]['annual_income'],
#         data[mask]['spending_score'],
#         label=f'Cluster {cluster}',
#         alpha=0.6
#     )
# plt.xlabel('Annual Income ($K)')
# plt.ylabel('Spending Score')
# plt.title('Customer Segments')
# plt.legend()
# plt.show()
```

### Task 3.4: Name Your Segments

Based on the cluster characteristics, give each segment a business-friendly name:

```python
# TODO: Create a segment naming dictionary
# Example:
# segment_names = {
#     0: "Budget Conscious",
#     1: "Young Professionals",
#     2: "Affluent Families",
#     ...
# }

# Explain your naming choices:
# Cluster 0: [Name] - [Reason based on characteristics]
# Cluster 1: [Name] - [Reason based on characteristics]
# ...
```

---

## Part 4: Business Recommendations (15 min)

### Task 4.1: Segment Profiles

Create a one-paragraph profile for each segment:

```python
# Segment 0: [Name]
# Profile: This segment consists of... They typically... Their behavior suggests...
# 
# Segment 1: [Name]
# Profile: ...
```

### Task 4.2: Marketing Recommendations

For each segment, recommend:
1. **Communication channel** (email, social media, direct mail)
2. **Offer type** (discounts, premium products, loyalty rewards)
3. **Messaging tone** (value-focused, luxury, family-oriented)

```python
# Segment 0: [Name]
# - Channel: 
# - Offer: 
# - Tone: 
#
# Segment 1: [Name]
# ...
```

### Task 4.3: High-Value Segment

Which segment should receive the most marketing investment? Why?

```python
# Answer:
```

---

## Bonus Challenges

### Challenge A: Silhouette Score

Use silhouette score to validate your choice of K:

```python
from sklearn.metrics import silhouette_score

# TODO: Calculate silhouette score for K from 2 to 10
# Which K has the highest silhouette score?
```

### Challenge B: 3D Visualization

Create a 3D scatter plot using income, spending, and age:

```python
from mpl_toolkits.mplot3d import Axes3D

# TODO: Create 3D visualization of clusters
```

### Challenge C: Predict New Customer Segment

A new customer has: income=$65K, spending_score=70, age=28, membership=2 years.
Which segment would they belong to?

```python
# TODO: Predict cluster for new customer
# new_customer = [[65, 70, 28, 2]]
# new_customer_scaled = scaler.transform(new_customer)
# predicted_cluster = kmeans.predict(new_customer_scaled)
```

---

## Definition of Done

- [ ] Data explored and visualized
- [ ] Elbow method applied and optimal K justified
- [ ] Clustering applied with chosen K
- [ ] All clusters analyzed with summary statistics
- [ ] Segments named with business-friendly labels
- [ ] At least 3 marketing recommendations provided
- [ ] High-value segment identified with justification

---

## Deliverables

1. **Python script** with all analysis code
2. **Visualization** of the elbow curve
3. **Visualization** of customer segments
4. **Written summary** (can be in code comments) with:
   - Segment names and profiles
   - Marketing recommendations
   - Business insights

---

## Grading Rubric

| Criterion | Points |
|-----------|--------|
| Correct K-Means implementation | 20 |
| Appropriate K selection with justification | 15 |
| Meaningful cluster interpretation | 25 |
| Business-relevant recommendations | 25 |
| Code quality and visualization | 15 |
| **Total** | **100** |

