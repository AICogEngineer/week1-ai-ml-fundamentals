"""
Demo 03: K-Means Clustering Visualization
==========================================
Week 1, Tuesday - AI/ML Fundamentals

This demo introduces UNSUPERVISED learning through K-Means clustering.
No labels! The algorithm discovers natural groupings in the data.

INSTRUCTOR NOTES:
- Key contrast: "Now we have NO LABELS - the algorithm finds structure itself"
- Step through the animation slowly to show centroid updates
- The elbow method is a practical skill they'll use often

Estimated Time: 20-25 minutes
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# SECTION 1: CREATE SYNTHETIC DATA
# =============================================================================
print("=" * 60)
print("DEMO 03: K-MEANS CLUSTERING")
print("Discovering Structure Without Labels")
print("=" * 60)

# Generate clustered data (we know the "truth" but pretend we don't)
np.random.seed(42)
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)

print("\n--- THE DATA ---")
print(f"Number of samples: {len(X)}")
print(f"Number of features: {X.shape[1]}")
print("\nIMPORTANT: In real unsupervised learning, we DON'T have labels!")
print("The algorithm must DISCOVER the groups on its own.")

# Visualize raw data (without showing the true labels)
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], alpha=0.6, edgecolors='black', linewidth=0.5)
plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('Feature 2', fontsize=12)
plt.title('Raw Data: Can You See Natural Groups?', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('03_raw_clusters.png', dpi=100)
plt.show()

print("\n[Saved: 03_raw_clusters.png]")
print("DISCUSSION: How many groups do you see? Where would you draw boundaries?")

# =============================================================================
# SECTION 2: K-MEANS ALGORITHM STEP BY STEP
# =============================================================================
print("\n--- K-MEANS ALGORITHM ---")
print("""
The K-Means Algorithm:
1. INITIALIZE: Place K centroids randomly
2. ASSIGN: Assign each point to nearest centroid
3. UPDATE: Move centroids to mean of assigned points
4. REPEAT: Until centroids stop moving
""")

# Manual step-by-step visualization
def visualize_kmeans_steps(X, n_clusters=4, n_steps=5):
    """Visualize K-Means step by step."""
    
    # Initialize centroids randomly
    np.random.seed(42)
    centroid_indices = np.random.choice(len(X), n_clusters, replace=False)
    centroids = X[centroid_indices].copy()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))
    
    for step in range(n_steps + 1):
        ax = axes[step]
        
        # Assign points to nearest centroid
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        
        # Plot points colored by cluster
        for i in range(n_clusters):
            mask = labels == i
            ax.scatter(X[mask, 0], X[mask, 1], c=[colors[i]], alpha=0.5, 
                      edgecolors='black', linewidth=0.3, s=40)
        
        # Plot centroids
        ax.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', 
                  s=200, edgecolors='white', linewidth=2, zorder=10)
        
        if step == 0:
            ax.set_title(f'Step {step}: Initial Random Centroids', fontsize=11)
        else:
            ax.set_title(f'Step {step}: After Update', fontsize=11)
        
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.grid(True, alpha=0.3)
        
        # Update centroids for next step
        if step < n_steps:
            for i in range(n_clusters):
                mask = labels == i
                if mask.sum() > 0:
                    centroids[i] = X[mask].mean(axis=0)
    
    plt.suptitle('K-Means Algorithm: Step-by-Step Visualization', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('03_kmeans_steps.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    return labels, centroids

print("\nVisualizing K-Means iterations...")
labels_manual, centroids_manual = visualize_kmeans_steps(X, n_clusters=4, n_steps=5)

print("[Saved: 03_kmeans_steps.png]")
print("\nWatch how centroids (X markers) move to the center of their clusters!")

# =============================================================================
# SECTION 3: USE SKLEARN K-MEANS
# =============================================================================
print("\n--- SKLEARN K-MEANS ---")

# Fit K-Means
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X)

print(f"Number of clusters: {kmeans.n_clusters}")
print(f"Number of iterations: {kmeans.n_iter_}")
print(f"Inertia (within-cluster sum of squares): {kmeans.inertia_:.2f}")

# Show centroids
print("\nCluster Centroids:")
print(f"{'Cluster':<10} {'Feature 1':<15} {'Feature 2':<15}")
print("-" * 40)
for i, centroid in enumerate(kmeans.cluster_centers_):
    print(f"{i:<10} {centroid[0]:<15.3f} {centroid[1]:<15.3f}")

# =============================================================================
# SECTION 4: VISUALIZE FINAL CLUSTERS
# =============================================================================
print("\n--- FINAL CLUSTERING ---")

plt.figure(figsize=(10, 6))
colors = plt.cm.Set1(np.linspace(0, 1, 4))

for i in range(4):
    mask = kmeans.labels_ == i
    plt.scatter(X[mask, 0], X[mask, 1], c=[colors[i]], alpha=0.6, 
               edgecolors='black', linewidth=0.3, s=50, label=f'Cluster {i}')

# Plot centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
           c='black', marker='X', s=200, edgecolors='white', linewidth=2,
           label='Centroids', zorder=10)

plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('Feature 2', fontsize=12)
plt.title('K-Means Clustering Result (K=4)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('03_final_clusters.png', dpi=100)
plt.show()

print("[Saved: 03_final_clusters.png]")

# =============================================================================
# SECTION 5: THE ELBOW METHOD
# =============================================================================
print("\n--- THE ELBOW METHOD ---")
print("How do we choose K? Use the Elbow Method!")

# Calculate inertia for different K values
K_range = range(1, 11)
inertias = []

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)
    print(f"K={k}: Inertia = {km.inertia_:.2f}")

# Plot elbow curve
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=10)
plt.xlabel('Number of Clusters (K)', fontsize=12)
plt.ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
plt.title('The Elbow Method: Finding Optimal K', fontsize=14)

# Mark the elbow
plt.annotate('Elbow\n(K=4)', xy=(4, inertias[3]), xytext=(5.5, inertias[3] + 100),
            fontsize=12, ha='center',
            arrowprops=dict(arrowstyle='->', color='red', lw=2))

plt.axvline(x=4, color='red', linestyle='--', alpha=0.7)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('03_elbow_method.png', dpi=100)
plt.show()

print("\n[Saved: 03_elbow_method.png]")
print("\nThe 'elbow' appears around K=4 - adding more clusters doesn't help much.")

# =============================================================================
# SECTION 6: CLUSTER ANALYSIS
# =============================================================================
print("\n--- CLUSTER ANALYSIS ---")

print("\nCluster Statistics:")
print(f"{'Cluster':<10} {'Size':<10} {'Mean F1':<12} {'Mean F2':<12}")
print("-" * 45)
for i in range(4):
    mask = kmeans.labels_ == i
    cluster_data = X[mask]
    print(f"{i:<10} {mask.sum():<10} {cluster_data[:, 0].mean():<12.3f} {cluster_data[:, 1].mean():<12.3f}")

# =============================================================================
# SECTION 7: REAL-WORLD EXAMPLE: CUSTOMER SEGMENTATION
# =============================================================================
print("\n--- REAL-WORLD APPLICATION ---")
print("Customer Segmentation Example")

# Create customer data
np.random.seed(42)
customers = np.vstack([
    np.random.normal([30, 80], [8, 15], 50),   # Young, high spenders
    np.random.normal([55, 70], [10, 20], 50),  # Middle-aged, moderate
    np.random.normal([25, 30], [7, 12], 50),   # Young, low spenders
    np.random.normal([60, 25], [8, 10], 50),   # Older, low spenders
])

print(f"\nCustomer data: {len(customers)} customers")
print("Features: Age, Annual Spending ($1000s)")

# Scale features
scaler = StandardScaler()
customers_scaled = scaler.fit_transform(customers)

# Find clusters
customer_km = KMeans(n_clusters=4, random_state=42, n_init=10)
customer_labels = customer_km.fit_predict(customers_scaled)

# Visualize
plt.figure(figsize=(10, 6))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
segment_names = ['Segment A', 'Segment B', 'Segment C', 'Segment D']

for i in range(4):
    mask = customer_labels == i
    plt.scatter(customers[mask, 0], customers[mask, 1], c=colors[i], 
               alpha=0.6, edgecolors='black', linewidth=0.3, s=60,
               label=segment_names[i])

plt.xlabel('Age', fontsize=12)
plt.ylabel('Annual Spending ($1000s)', fontsize=12)
plt.title('Customer Segmentation with K-Means', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('03_customer_segments.png', dpi=100)
plt.show()

print("[Saved: 03_customer_segments.png]")

# Segment analysis
print("\nCustomer Segments Discovered:")
print(f"{'Segment':<12} {'Count':<8} {'Avg Age':<12} {'Avg Spend ($K)':<15}")
print("-" * 50)
for i in range(4):
    mask = customer_labels == i
    segment = customers[mask]
    print(f"{segment_names[i]:<12} {mask.sum():<8} {segment[:, 0].mean():<12.1f} ${segment[:, 1].mean():<14.1f}")

# =============================================================================
# SECTION 8: KEY TAKEAWAYS
# =============================================================================
print("\n" + "=" * 60)
print("KEY TAKEAWAYS")
print("=" * 60)
print("""
1. UNSUPERVISED vs SUPERVISED:
   - Supervised: We have labels, predict outcomes
   - Unsupervised: NO labels, discover structure

2. K-MEANS ALGORITHM:
   - Initialize K centroids
   - Assign points to nearest centroid
   - Update centroids to cluster means
   - Repeat until convergence

3. CHOOSING K:
   - Use the Elbow Method
   - Look for the "bend" where improvement slows
   - Domain knowledge also helps!

4. APPLICATIONS:
   - Customer segmentation
   - Document clustering
   - Image compression
   - Anomaly detection

5. LIMITATIONS:
   - Must specify K in advance
   - Assumes spherical clusters
   - Sensitive to initialization (use n_init)
""")

print("Today's Journey Complete!")
print("Supervised (Regression, Classification) -> Unsupervised (Clustering)")

