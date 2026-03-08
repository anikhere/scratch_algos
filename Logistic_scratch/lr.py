import numpy as np
X = np.array([
    [2,3],
    [1,2],
    [2,1],
    [3,5],
    [2,4]
])

y = np.array([0,0,0,1,1])
class LR:
    def __init__(self,iterations=100,lr=0.05,w=np.array([1,2]),b=0.3,threshold=0.5):
        self.iter = iterations
        self.lr = lr
        self.w = w
        self.b = b
        self.threshold= threshold
    
    def sigmoid(self,y):
        sig = 1/(1+np.exp(-y))
        return sig 
    
    def general(self,x:np.array):
        z = x @ self.w + self.b
        p = self.sigmoid(z)
        return p
    
    def Gradient(self,x:np.array,y:np.array):
        shape_x = x.shape[0]
        after_sig = self.general(x)
        error = after_sig - y
        dw = (1/shape_x)*(x.T @ error)
        db = (1/shape_x)*(np.sum(error))
        return dw,db
    
    def fit(self,x,y):
        for iters in range(self.iter):
            dw,db = self.Gradient(x,y)
            self.w = self.w - self.lr*dw
            self.b = self.b - self.lr*db 
            if iters%10==0:
                print(f'the loss is {self.loss(x=x,y=y)}')
            
    
    def loss(self,x,y):
        p = self.general(x)
        loss = - np.mean(y*np.log(p) + (1-y)*np.log(1-p))
        return loss

    
    def predict(self, val):   
        p = self.general(val)
        print(f'the probablities are {p}')
        preds = (p >= self.threshold).astype(int)
        return preds        
        


