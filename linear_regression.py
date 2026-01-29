class linear_Regression:
    def __init__(self, iteration=1000, learning_rate=0.01):
        self.iteration = iteration
        self.learning_rate = learning_rate
        self.b = 0
        self.m = 0
    
    def predict(self,x):
      n = len(x)
      results = []
      for i in range(n):
        y = self.m*x[i] + self.b
        results.append(y)
      return results
    
    def mse(self,true,pred):
      n = len(true)
      mean = 0

      for i in range(n):
        diff = pred[i] - true[i]
        squared = diff**2
        mean+=squared
      return mean/n
    
    def fit(self,x,y):
      n = len(x)
      for iteration in range(self.iteration):
        y_pred = self.predict(x)
        dm =0
        db=0
        for i in range(n):
          dm += x[i] * (y[i] - y_pred[i])
          db += (y[i] - y_pred[i])
        dm = (-2/n) * dm
        db = (-2/n) * db
        
        self.m = self.m - self.learning_rate * dm  # ✅ Subtract!
        self.b = self.b - self.learning_rate * db  # ✅ Subtract!
        if iteration % 100 == 0:
          current_mse = self.mse(y, y_pred)
          print(f"Iter {iteration}: MSE={current_mse:.4f}, m={self.m:.4f}, b={self.b:.4f}")
          
# Simple dataset: y = 2x
X = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Train model
model = linear_Regression(iteration=1000, learning_rate=0.01)
model.fit(X, y)

print(f"\nFinal m: {model.m:.4f} (should be close to 2.0)")
print(f"Final b: {model.b:.4f} (should be close to 0.0)")

# Test predictions
X_test = [6, 7, 8]
predictions = model.predict(X_test)
print(f"\nPredictions for {X_test}: {predictions}")
print(f"Expected: [12, 14, 16]")          
          