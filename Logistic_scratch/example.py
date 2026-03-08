import numpy as np
from lr import LR
# training data
X = np.array([
    [2,3],
    [1,2],
    [2,1],
    [3,5],
    [2,4]
])

y = np.array([0,0,0,1,1])
model = LR(iterations=100,lr=0.05,threshold=0.5)
model.fit(x=X,y=y)
print(f'the loss is {model.loss(x=X,y=y)}')
print('weights',model.w)
print('bias',model.b)
test_data = np.array([
    [2,2],
    [3,4],
    [0,1]
])
print(f'the prediction is {model.predict(test_data)}')