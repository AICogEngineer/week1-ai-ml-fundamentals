# Weekly Knowledge Check: ML Fundamentals (Tuesday)

## Part 1: Multiple Choice

### 1. What distinguishes machine learning from traditional programming?

- [ ] A) Machine learning uses faster computers
- [ ] B) Machine learning extracts rules from data rather than following hand-coded rules
- [ ] C) Machine learning only works with numbers
- [ ] D) Machine learning requires the internet

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Machine learning extracts rules from data rather than following hand-coded rules

**Explanation:** Traditional programming follows the paradigm: Input Data + Rules --> Output. Machine learning inverts this: Input Data + Output --> Rules (Model). The algorithm discovers patterns from examples rather than following explicitly programmed logic.

- **Why others are wrong:**
  - A) Speed is not the defining characteristic
  - C) ML works with various data types including text and images
  - D) ML can run entirely offline
</details>

---

### 2. Which learning paradigm requires labeled data?

- [ ] A) Unsupervised learning
- [ ] B) Reinforcement learning
- [ ] C) Supervised learning
- [ ] D) Transfer learning

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) Supervised learning

**Explanation:** Supervised learning trains on labeled examples (input-output pairs). The model learns by comparing its predictions to the known correct answers and adjusting accordingly.

- **Why others are wrong:**
  - A) Unsupervised learning works with unlabeled data to discover patterns
  - B) Reinforcement learning uses reward signals, not labels
  - D) Transfer learning is a technique, not a paradigm defined by labels
</details>

---

### 3. You need to predict house prices based on features like square footage, bedrooms, and age. Which type of supervised learning problem is this?

- [ ] A) Classification
- [ ] B) Clustering
- [ ] C) Regression
- [ ] D) Dimensionality reduction

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) Regression

**Explanation:** Regression predicts continuous numerical values. House prices are continuous ($150,000, $247,500, etc.), not discrete categories.

- **Why others are wrong:**
  - A) Classification predicts discrete categories (spam/not spam, cat/dog)
  - B) Clustering is unsupervised, not for prediction
  - D) Dimensionality reduction compresses features, doesn't predict values
</details>

---

### 4. In the linear regression equation y = wx + b, what does 'b' represent?

- [ ] A) The weight
- [ ] B) The input feature
- [ ] C) The bias (y-intercept)
- [ ] D) The prediction error

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) The bias (y-intercept)

**Explanation:** In y = wx + b, 'b' is the bias term (also called the intercept). It represents the predicted value when all inputs are zero, allowing the regression line to shift up or down from the origin.

- **Why others are wrong:**
  - A) 'w' is the weight (slope)
  - B) 'x' is the input feature
  - D) Prediction error is y_actual - y_predicted, not part of the equation
</details>

---

### 5. What is the primary purpose of Mean Squared Error (MSE)?

- [ ] A) To normalize the data
- [ ] B) To measure prediction error by squaring differences between actual and predicted values
- [ ] C) To classify data into categories
- [ ] D) To reduce the number of features

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) To measure prediction error by squaring differences between actual and predicted values

**Explanation:** MSE = (1/n) * SUM[(actual - predicted)^2]. Squaring makes all errors positive, penalizes large errors more than small ones, and is mathematically convenient (differentiable).

- **Why others are wrong:**
  - A) Normalization is a separate preprocessing step
  - C) MSE is for regression, not classification
  - D) Feature reduction uses techniques like PCA, not MSE
</details>

---

### 6. Why is the MSE squared rather than just using absolute differences?

- [ ] A) It's computationally faster
- [ ] B) It prevents cancellation of positive/negative errors and penalizes large errors more
- [ ] C) It makes the values smaller
- [ ] D) It converts errors to probabilities

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) It prevents cancellation of positive/negative errors and penalizes large errors more

**Explanation:** Without squaring, positive errors (+5) and negative errors (-5) would cancel out, giving a misleading average of 0. Squaring ensures all contributions are positive and makes large errors (10^2=100) more impactful than small errors (1^2=1).

- **Why others are wrong:**
  - A) Squaring is slightly more expensive than absolute value
  - C) Squaring makes values larger (10^2 = 100)
  - D) MSE doesn't produce probabilities
</details>

---

### 7. What does a classification algorithm learn?

- [ ] A) How to predict continuous numbers
- [ ] B) Decision boundaries that separate different categories
- [ ] C) How to cluster unlabeled data
- [ ] D) The average of all features

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Decision boundaries that separate different categories

**Explanation:** Classification algorithms learn to draw boundaries (lines, curves, or hyperplanes) that separate data points belonging to different classes. Points on one side of the boundary are predicted as one class, points on the other side as another class.

- **Why others are wrong:**
  - A) Continuous prediction is regression
  - C) Clustering unlabeled data is unsupervised learning
  - D) Averaging features is not classification
</details>

---

### 8. Despite having "regression" in its name, logistic regression is actually a:

- [ ] A) Regression algorithm
- [ ] B) Clustering algorithm
- [ ] C) Classification algorithm
- [ ] D) Dimensionality reduction algorithm

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) Classification algorithm

**Explanation:** Logistic regression outputs probabilities (values between 0 and 1) using the sigmoid function, which are then thresholded to make discrete class predictions. The "regression" part refers to the underlying linear model before the sigmoid transformation.

- **Why others are wrong:**
  - A) It predicts categories, not continuous values
  - B) It uses labeled data, not unlabeled clustering
  - D) It doesn't reduce dimensions
</details>

---

### 9. What problem does the sigmoid function solve in logistic regression?

- [ ] A) It speeds up computation
- [ ] B) It bounds outputs between 0 and 1, interpretable as probabilities
- [ ] C) It removes outliers
- [ ] D) It normalizes the input features

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) It bounds outputs between 0 and 1, interpretable as probabilities

**Explanation:** Linear regression outputs unbounded values (could be -5 or 100), which don't make sense as probabilities. The sigmoid function S(z) = 1/(1+e^-z) squashes any input to a value strictly between 0 and 1.

- **Why others are wrong:**
  - A) Sigmoid involves exponential computation, not faster
  - C) Outlier removal is a separate preprocessing step
  - D) Feature normalization happens before the model
</details>

---

### 10. K-Means clustering requires you to specify:

- [ ] A) The target labels for each cluster
- [ ] B) The number of clusters (K) in advance
- [ ] C) The exact position of each centroid
- [ ] D) The color of each cluster

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) The number of clusters (K) in advance

**Explanation:** K-Means requires you to choose K before running the algorithm. This is a key limitation - you must decide how many groups you want, even if the "natural" number of clusters is unknown.

- **Why others are wrong:**
  - A) K-Means is unsupervised - no labels provided
  - C) Centroids are randomly initialized, then refined
  - D) Colors are just visualization, not algorithm input
</details>

---

### 11. What is the correct order of steps in the K-Means algorithm?

- [ ] A) Update centroids, Assign points, Initialize centroids
- [ ] B) Assign points, Initialize centroids, Update centroids
- [ ] C) Initialize centroids, Assign points to nearest centroid, Update centroids to cluster means
- [ ] D) Update centroids, Initialize centroids, Assign points

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) Initialize centroids, Assign points to nearest centroid, Update centroids to cluster means

**Explanation:** K-Means follows this loop: (1) Initialize K centroids randomly, (2) Assign each point to its nearest centroid, (3) Update each centroid to the mean of its assigned points, (4) Repeat steps 2-3 until convergence.

- **Why others are wrong:**
  - A, B, D) These have steps in wrong order - can't assign to centroids before they exist
</details>

---

### 12. The Elbow Method helps determine:

- [ ] A) Which features to use
- [ ] B) The optimal number of clusters (K)
- [ ] C) Whether to use supervised or unsupervised learning
- [ ] D) The learning rate

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) The optimal number of clusters (K)

**Explanation:** The Elbow Method plots Within-Cluster Sum of Squares (WCSS) against different K values. The "elbow" point where additional clusters stop significantly reducing WCSS suggests the optimal K.

- **Why others are wrong:**
  - A) Feature selection uses different techniques
  - C) Problem type is determined by data structure
  - D) Learning rate is for gradient descent, not clustering
</details>

---

### 13. Euclidean distance between points (1,2) and (4,6) is:

- [ ] A) 7
- [ ] B) 5
- [ ] C) 25
- [ ] D) 3

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) 5

**Explanation:** Euclidean distance = sqrt[(4-1)^2 + (6-2)^2] = sqrt[9 + 16] = sqrt[25] = 5

- **Why others are wrong:**
  - A) 7 would be the Manhattan distance: |4-1| + |6-2| = 3 + 4 = 7
  - C) 25 is the squared distance before taking the square root
  - D) 3 is only the difference in one dimension
</details>

---

### 14. Manhattan distance between points (1,2) and (4,6) is:

- [ ] A) 7
- [ ] B) 5
- [ ] C) 25
- [ ] D) 3

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** A) 7

**Explanation:** Manhattan distance = |4-1| + |6-2| = 3 + 4 = 7. It sums the absolute differences along each dimension, like walking city blocks.

- **Why others are wrong:**
  - B) 5 is the Euclidean (straight-line) distance
  - C) 25 is Euclidean distance squared
  - D) 3 is only one component of the sum
</details>

---

### 15. Why is feature scaling important for distance-based algorithms like K-Means?

- [ ] A) It makes the algorithm run faster
- [ ] B) It prevents features with larger ranges from dominating the distance calculation
- [ ] C) It removes outliers from the data
- [ ] D) It converts categorical features to numerical

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) It prevents features with larger ranges from dominating the distance calculation

**Explanation:** If income ranges from 0-100,000 and age ranges from 0-100, the income dimension will completely dominate distance calculations. Scaling (e.g., StandardScaler) puts all features on comparable scales.

- **Why others are wrong:**
  - A) Scaling doesn't significantly affect speed
  - C) Outlier removal is separate from scaling
  - D) Encoding handles categorical features, not scaling
</details>

---

## Part 2: True/False

### 16. In supervised learning, the model learns from data where the correct answers are already known.

- [ ] True
- [ ] False

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** True

**Explanation:** This is the defining characteristic of supervised learning. The "supervision" comes from labeled examples where we know the correct output for each input.
</details>

---

### 17. Unsupervised learning can be used to predict specific target values.

- [ ] True
- [ ] False

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** False

**Explanation:** Unsupervised learning discovers patterns and structure in data WITHOUT predicting specific targets. It groups similar items, reduces dimensions, or finds anomalies - but doesn't make predictions in the supervised sense.
</details>

---

### 18. A single perceptron can solve the XOR problem.

- [ ] True
- [ ] False

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** False

**Explanation:** XOR is not linearly separable - no single straight line can separate the classes. This limitation, proven by Minsky and Papert in 1969, requires multi-layer networks to solve.
</details>

---

### 19. K-Means clustering always finds the same clusters regardless of initial centroid positions.

- [ ] True
- [ ] False

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** False

**Explanation:** K-Means is sensitive to initialization. Different random starting positions can lead to different final clusters. This is why K-Means++ initialization and running multiple times with different seeds are recommended practices.
</details>

---

### 20. The cost function in machine learning measures how well the model fits the training data.

- [ ] True
- [ ] False

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** True

**Explanation:** The cost function (or loss function) quantifies the difference between predictions and actual values. Lower cost = better fit. The model learns by minimizing this cost.
</details>

---

## Part 3: Code Prediction

### 21. What is the output of this code?

```python
from sklearn.cluster import KMeans
import numpy as np

X = np.array([[1, 1], [1, 2], [10, 10], [10, 11]])
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(X)
print(len(np.unique(kmeans.labels_)))
```

- [ ] A) 1
- [ ] B) 2
- [ ] C) 4
- [ ] D) 0

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) 2

**Explanation:** We specified `n_clusters=2`, so K-Means will assign each point to one of 2 clusters. `kmeans.labels_` contains the cluster assignment (0 or 1) for each point. `np.unique()` finds distinct values, which will be [0, 1], and `len()` of that is 2.
</details>

---

### 22. What does this code compute?

```python
import numpy as np

actual = np.array([3, 5, 7])
predicted = np.array([2.5, 5.5, 6])
result = np.mean((actual - predicted) ** 2)
print(result)
```

- [ ] A) Mean Absolute Error
- [ ] B) Mean Squared Error
- [ ] C) Root Mean Squared Error
- [ ] D) R-squared

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Mean Squared Error

**Explanation:** The formula `np.mean((actual - predicted) ** 2)` is exactly the MSE formula: average of squared differences. The result is (0.25 + 0.25 + 1) / 3 = 0.5.

- **Why others are wrong:**
  - A) MAE would use `np.abs()` instead of squaring
  - C) RMSE would take `np.sqrt()` of MSE
  - D) R-squared is a different formula involving variance
</details>

---

### 23. What will `prediction` be after this code runs?

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

X = np.array([[0], [1], [2], [3]])
y = np.array([0, 0, 1, 1])
model = LogisticRegression()
model.fit(X, y)
prediction = model.predict([[1.5]])[0]
```

- [ ] A) 0.5
- [ ] B) 0
- [ ] C) 1
- [ ] D) 1.5

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) 1

**Explanation:** The data shows 0,1 map to class 0 and 2,3 map to class 1. Input 1.5 is on the boundary but will be classified. `predict()` returns class labels (0 or 1), not probabilities. Given the training data pattern, 1.5 will likely be classified as 1.

- **Why others are wrong:**
  - A) 0.5 would be a probability, but `predict()` returns class labels
  - B) Possible but less likely given 1.5 is closer to the "1" region
  - D) 1.5 is the input, not a valid class output
</details>

---

### 24. What is the shape of `distances` after this code?

```python
from sklearn.metrics import pairwise_distances
import numpy as np

X = np.array([[1, 2], [3, 4], [5, 6]])
distances = pairwise_distances(X)
print(distances.shape)
```

- [ ] A) (3,)
- [ ] B) (3, 2)
- [ ] C) (3, 3)
- [ ] D) (2, 3)

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) (3, 3)

**Explanation:** `pairwise_distances(X)` computes the distance from every point to every other point. With 3 points, this creates a 3x3 matrix where entry [i,j] is the distance from point i to point j.
</details>

---

## Part 4: Fill-in-the-Blank

### 25. Machine learning follows the paradigm: Data + _______ = Model

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** Algorithm

**Explanation:** The ML equation is: Training Data + Learning Algorithm = Trained Model. The algorithm processes the data to produce a model that can make predictions.
</details>

---

### 26. In the decision flowchart for ML problems, if you have labeled data and want to predict categories, you should use _______.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** Classification (Supervised Learning)

**Explanation:** Labeled data + category prediction = classification. This is one of the two main supervised learning tasks (the other being regression for continuous values).
</details>

---

### 27. The formula for linear regression is y = wx + _______.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** b (bias/intercept)

**Explanation:** y = wx + b, where w is the weight (slope), x is the input, and b is the bias (intercept) that shifts the line up or down.
</details>

---

### 28. K-Means clustering uses _______ as the center of each cluster.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** Centroids (or means)

**Explanation:** The "Means" in K-Means refers to centroids, which are the mean (average) positions of all points assigned to that cluster.
</details>

---

### 29. To calculate Euclidean distance in 2D, you take the square root of the sum of squared _______.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** Differences (or differences between coordinates)

**Explanation:** Euclidean distance = sqrt[(x2-x1)^2 + (y2-y1)^2]. You square the differences in each dimension, sum them, then take the square root.
</details>

---

### 30. R-squared (R^2) measures the proportion of _______ explained by the model.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** Variance

**Explanation:** R^2 indicates how much of the variance in the target variable is explained by the model's predictions. R^2 = 0.95 means the model explains 95% of the variance.
</details>

---

## Part 5: Scenario-Based Questions

### 31. A hospital wants to predict whether a tumor is malignant or benign based on cell measurements. Which approach should they use?

- [ ] A) K-Means clustering
- [ ] B) Linear regression
- [ ] C) Classification (e.g., Logistic Regression)
- [ ] D) Dimensionality reduction

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) Classification (e.g., Logistic Regression)

**Explanation:** This is a binary classification problem: predicting one of two categories (malignant/benign) from features. The hospital has historical labeled data (pathology results), making this supervised classification.

- **Why others are wrong:**
  - A) Clustering doesn't make predictions for specific outcomes
  - B) Regression predicts continuous values, not categories
  - D) Dimensionality reduction compresses features, doesn't predict
</details>

---

### 32. A marketing team has customer purchase data but no predefined segments. They want to discover natural customer groups. Which approach should they use?

- [ ] A) Supervised classification
- [ ] B) Linear regression
- [ ] C) Unsupervised clustering (e.g., K-Means)
- [ ] D) Logistic regression

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) Unsupervised clustering (e.g., K-Means)

**Explanation:** With no predefined labels and the goal of discovering natural groupings, this is a textbook unsupervised learning scenario. K-Means can segment customers based on behavior patterns without needing prior labels.

- **Why others are wrong:**
  - A, D) Supervised methods require labeled data
  - B) Regression predicts values, doesn't discover groups
</details>

---

### 33. Your model achieves 99% accuracy on a fraud detection task where 99% of transactions are legitimate. Is this good performance?

- [ ] A) Yes, 99% is excellent
- [ ] B) No, the model might just be predicting "legitimate" for everything
- [ ] C) Yes, but only if precision is also high
- [ ] D) Cannot determine without the dataset size

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) No, the model might just be predicting "legitimate" for everything

**Explanation:** With 99% legitimate transactions, a model that always predicts "legitimate" achieves 99% accuracy while catching zero fraud! This is the class imbalance problem. You need precision, recall, and F1-score to evaluate properly.
</details>

---

### 34. You're building a model to predict stock prices. The model performs perfectly on historical data but poorly on recent data. What is this symptom called?

- [ ] A) Underfitting
- [ ] B) Overfitting
- [ ] C) Feature engineering
- [ ] D) Cross-validation

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Overfitting

**Explanation:** Overfitting occurs when a model memorizes training data patterns (including noise) instead of learning generalizable patterns. It performs well on training data but poorly on new, unseen data.

- **Why others are wrong:**
  - A) Underfitting shows poor performance on BOTH training and test data
  - C) Feature engineering is creating/selecting features
  - D) Cross-validation is an evaluation technique
</details>

---

### 35. When should you NOT use machine learning?

- [ ] A) When you have lots of data
- [ ] B) When simple rules suffice to solve the problem
- [ ] C) When the problem requires personalization
- [ ] D) When rules change frequently

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) When simple rules suffice to solve the problem

**Explanation:** ML adds complexity. If you can solve a problem with simple if-else rules (e.g., "reject if amount > $10,000 and account age < 1 day"), that's simpler, faster, more interpretable, and easier to maintain than an ML model.

- **Why others are wrong:**
  - A, C, D) These are all scenarios where ML shines
</details>

---

## Bonus Questions

### 36. What is the "curse of dimensionality" in the context of distance metrics?

- [ ] A) High dimensions make computers crash
- [ ] B) In high dimensions, all points become roughly equidistant, making distance less meaningful
- [ ] C) Distance can only be computed in 3 dimensions
- [ ] D) Higher dimensions require more memory

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) In high dimensions, all points become roughly equidistant, making distance less meaningful

**Explanation:** As dimensions increase, the ratio of nearest to farthest distances approaches 1. This means "nearest neighbor" loses meaning because everything is approximately the same distance away.
</details>

---

### 37. What is K-Means++ and why is it used?

- [ ] A) A faster version of K-Means
- [ ] B) An initialization strategy that spreads initial centroids apart for better convergence
- [ ] C) A method to automatically determine K
- [ ] D) A variant that allows clusters to overlap

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) An initialization strategy that spreads initial centroids apart for better convergence

**Explanation:** K-Means++ chooses initial centroids probabilistically, with points farther from existing centroids more likely to be selected. This spreads centroids apart, leading to better and more consistent clustering results than random initialization.
</details>

---

### 38. In the confusion matrix for binary classification, what does a False Positive mean?

- [ ] A) Model correctly predicted positive class
- [ ] B) Model predicted positive but actual was negative
- [ ] C) Model predicted negative but actual was positive
- [ ] D) Model correctly predicted negative class

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Model predicted positive but actual was negative

**Explanation:** False Positive = "falsely positive" = the model said positive/yes, but it was wrong - the actual value was negative. Example: Spam filter marking a legitimate email as spam.

- **Why others are wrong:**
  - A) That's a True Positive
  - C) That's a False Negative
  - D) That's a True Negative
</details>

---

### 39. When would Manhattan distance be preferred over Euclidean distance?

- [ ] A) When working with geographic coordinates
- [ ] B) When working with high-dimensional or sparse data
- [ ] C) When you want to emphasize large differences
- [ ] D) When computing shortest paths

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) When working with high-dimensional or sparse data

**Explanation:** Manhattan distance is more robust in high dimensions because it doesn't square differences (which would make one large difference dominate). It treats all dimensions more equally and is often preferred for text/document similarity.

- **Why others are wrong:**
  - A) Geographic coordinates typically use Euclidean or great-circle distance
  - C) Euclidean emphasizes large differences due to squaring
  - D) Shortest paths depend on the graph structure, not the metric
</details>

---

### 40. What is the relationship between precision and recall in the context of the classification threshold?

- [ ] A) Increasing threshold increases both precision and recall
- [ ] B) There is a trade-off: higher threshold increases precision but decreases recall
- [ ] C) Threshold has no effect on precision and recall
- [ ] D) Lower threshold always improves model performance

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) There is a trade-off: higher threshold increases precision but decreases recall

**Explanation:** Higher threshold means "only predict positive if very confident" --> fewer false positives (higher precision) but more false negatives (lower recall). Lower threshold catches more positives (higher recall) but includes more mistakes (lower precision).
</details>

---

*Quiz generated by Practice Quiz Agent for Week 1: AI/ML Fundamentals - Tuesday Content*

