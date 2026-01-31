# KNN from Scratch (Python)

## Overview
This project implements the K-Nearest Neighbors algorithm from scratch
without using sklearn.

## Algorithm Steps
1. Compute Euclidean distance
2. Sort distances
3. Select k nearest neighbors
4. Majority voting

## Time Complexity
- Prediction: O(n * d)

## Example
```python
knn = KNN(k=3, features=X_train, labels=y_train)
knn.predict([5.1, 3.5, 1.4, 0.2])
