# kNN implementation from scratch
from collections import Counter
import math
class KNN:
    def __init__(self,k,labels,features):
        self.k = k
        self.labels = labels
        self.features = features
    def distance(self,a,b):
        total=0
        for i in range(len(a)):
            total+=(a[i] - b[i])**2
        return math.sqrt(total)
    def predict(self,sample):
        distance = []
        for i in range(len(self.features)):
           dis = self.distance(self.features[i],sample)
           distance.append((dis,self.labels[i]))
        print(f'the distance is {distance}')
        distance.sort(key=lambda x:x[0])
        neighbors = distance[:self.k]
        print(f'the neighbors are {neighbors}')
        n_labels = [labels for dis,labels in neighbors]
        most_common = Counter(n_labels).most_common(1)
        prediction = most_common[0][0]
        return prediction


