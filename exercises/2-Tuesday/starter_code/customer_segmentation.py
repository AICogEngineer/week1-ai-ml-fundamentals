"""
Exercise 03: K-Means Customer Segmentation
==========================================

Apply K-Means clustering to discover customer segments.
Complete the TODO sections.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# =============================================================================
# PART 1: DATA EXPLORATION
# =============================================================================

print("=" * 60)
print("CUSTOMER SEGMENTATION WITH K-MEANS")
print("=" * 60)

# Generate realistic customer data
np.random.seed(42)
n_customers = 500

data = pd.DataFrame({
    'customer_id': range(1, n_customers + 1),
    'annual_income': np.concatenate([
        np.random.normal(30, 8, 100),
        np.random.normal(55, 10, 150),
        np.random.normal(90, 12, 150),
        np.random.normal(120, 15, 100)
    ]),
    'spending_score': np.concatenate([
        np.random.normal(25, 10, 100),
        np.random.normal(50, 15, 150),
        np.random.normal(75, 12, 150),
        np.random.normal(45, 20, 100)
    ]),
    'age': np.concatenate([
        np.random.normal(22, 3, 100),
        np.random.normal(35, 8, 150),
        np.random.normal(45, 10, 150),
        np.random.normal(55, 8, 100)
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

print("\n--- DATA OVERVIEW ---")
# TODO: Explore the data
# print(data.head(10))
# print("\nStatistics:")
# print(data.describe())

# TODO: Visualize relationships
# plt.figure(figsize=(15, 5))
# 
# plt.subplot(1, 3, 1)
# plt.scatter(data['annual_income'], data['spending_score'], alpha=0.5)
# plt.xlabel('Annual Income ($K)')
# plt.ylabel('Spending Score')
# plt.title('Income vs Spending')
# 
# plt.subplot(1, 3, 2)
# plt.scatter(data['age'], data['spending_score'], alpha=0.5)
# plt.xlabel('Age')
# plt.ylabel('Spending Score')
# plt.title('Age vs Spending')
# 
# plt.subplot(1, 3, 3)
# plt.scatter(data['annual_income'], data['membership_years'], alpha=0.5)
# plt.xlabel('Annual Income ($K)')
# plt.ylabel('Membership Years')
# plt.title('Income vs Membership')
# 
# plt.tight_layout()
# plt.show()

# =============================================================================
# PART 2: FINDING OPTIMAL K
# =============================================================================

print("\n--- FINDING OPTIMAL K ---")

# Select features for clustering
features = ['annual_income', 'spending_score', 'age', 'membership_years']
X = data[features].values

# TODO: Scale the features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# TODO: Elbow method
# K_range = range(1, 11)
# inertias = []
# 
# for k in K_range:
#     kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
#     kmeans.fit(X_scaled)
#     inertias.append(kmeans.inertia_)
#     print(f"K={k}: Inertia = {kmeans.inertia_:.2f}")

# TODO: Plot elbow curve
# plt.figure(figsize=(10, 6))
# plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=10)
# plt.xlabel('Number of Clusters (K)')
# plt.ylabel('Inertia (Within-Cluster Sum of Squares)')
# plt.title('Elbow Method for Optimal K')
# plt.grid(True, alpha=0.3)
# plt.show()

# TODO: Choose optimal K
optimal_k = None  # Your answer based on elbow plot

# Justification for your choice:
# 

# =============================================================================
# PART 3: CLUSTERING AND INTERPRETATION
# =============================================================================

print("\n--- APPLYING K-MEANS ---")

# TODO: Fit K-Means
# kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
# data['cluster'] = kmeans.fit_predict(X_scaled)

# TODO: Analyze clusters
# print("\nCluster Summary (Mean Values):")
# cluster_summary = data.groupby('cluster')[features].mean()
# print(cluster_summary.round(2))

# print("\nCustomers per Cluster:")
# print(data['cluster'].value_counts().sort_index())

# TODO: Visualize clusters
# plt.figure(figsize=(12, 8))
# colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
# 
# for cluster in range(optimal_k):
#     mask = data['cluster'] == cluster
#     plt.scatter(
#         data[mask]['annual_income'],
#         data[mask]['spending_score'],
#         c=colors[cluster],
#         label=f'Cluster {cluster}',
#         alpha=0.6,
#         edgecolors='black',
#         linewidth=0.5
#     )
# 
# plt.xlabel('Annual Income ($K)', fontsize=12)
# plt.ylabel('Spending Score', fontsize=12)
# plt.title('Customer Segments', fontsize=14)
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.show()

# =============================================================================
# PART 4: SEGMENT NAMING AND BUSINESS RECOMMENDATIONS
# =============================================================================

print("\n--- SEGMENT ANALYSIS ---")

# TODO: Name your segments based on their characteristics
segment_names = {
    # 0: "???",
    # 1: "???",
    # 2: "???",
    # ...
}

# TODO: Create segment profiles
# For each segment, describe:
# - Demographics (age, income)
# - Behavior (spending, membership)
# - What makes them unique

# Segment 0: [Name]
# Profile: 

# Segment 1: [Name]
# Profile: 

# TODO: Marketing recommendations for each segment
# Channel, Offer Type, Messaging Tone

# Segment 0 Recommendations:
# - Channel: 
# - Offer: 
# - Tone: 

# TODO: Identify highest-value segment
# Which segment deserves the most marketing investment?
# Answer: 
# Justification: 

# =============================================================================
# BONUS CHALLENGES
# =============================================================================

# BONUS A: Silhouette Score
# from sklearn.metrics import silhouette_score
# TODO: Calculate silhouette scores for K from 2 to 10

# BONUS B: 3D Visualization
# from mpl_toolkits.mplot3d import Axes3D
# TODO: Create 3D scatter plot

# BONUS C: Predict new customer segment
# new_customer = [[65, 70, 28, 2]]  # income, spending, age, membership
# TODO: Predict which cluster this customer belongs to

