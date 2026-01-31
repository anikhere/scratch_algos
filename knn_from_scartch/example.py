from knn import KNN

# Training data (features)
X_train = [
    [1, 2],
    [2, 3],
    [3, 3],
    [6, 5],
    [7, 7]
]

# Labels
y_train = [0, 0, 0, 1, 1]

# Create model
knn = KNN(k=3, features=X_train, labels=y_train)

# Test sample
test_sample = [3, 2]

prediction = knn.predict(test_sample)
print("Predicted class:", prediction)
