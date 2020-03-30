import numpy as np
import math

class MKNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y
        return self

    def euclidean_distance(self, row1, row2):
        distance = 0.0
        for i in range(len(row1)-1):
            distance += (row1[i] - row2[i])**2
        return np.sqrt(distance)
 
    # Locate the most similar neighbors
    def get_neighbors(self, test_row, k):
        distances = list()
        for (trainrow,label) in zip(self.X, self.y):
            dist = self.euclidean_distance(test_row, trainrow)
            distances.append((trainrow, dist, label))
        distances.sort(key=lambda tup: tup[1])
        neighbors_label = list()
        neighbors = list()
        for i in range(k):
            neighbors.append(distances[i][0])
            neighbors_label.append(distances[i][2])
        return neighbors,neighbors_label
        
    def k_nearest_neighbors(self, X_test, k):
        neighbors_label = list()
        neighbors = list()
        for row in X_test:
            neigh, label = self.get_neighbors(row, k)
            neighbors_label.append(label)
            neighbors.append(neigh)
        return(neighbors, neighbors_label)
        
    def k_nearest_neighbors_label_count(self, X_test, k):
        list_count = []
        (neighbors, neighbors_label) = self.k_nearest_neighbors(X_test,k)
        neighbors_label = np.asarray(neighbors_label)
        for neigh_row in neighbors_label:
            (unique, counts) = np.unique(neigh_row, return_counts=True)
            freq = np.asarray((unique, counts)).T
            list_count.append(dict(freq.tolist()))
            
        return list_count, neighbors, neighbors_label
        
    def marginalised_knn(self, X_test, k):
    
        # get neighbor pairs
        list_countA, neighborsA, neighbors_labelA = self.k_nearest_neighbors_label_count(X_test,k)
        
        # calculate marginalised probability for each pairs
        predicted_label, predicted_probab = [],[]
        for listA,neighbor,labelA in zip(list_countA,neighborsA,neighbors_labelA):
            list_countB, neighborsB, neighbors_labelB = self.k_nearest_neighbors_label_count(neighbor,k)
            pairs_count = []
            for listB in (list_countB):
                mul = 0
                keys = listA.keys()
                for i in keys:
                    if(i in listB):
                        mul = mul + (listA.get(i)*listB.get(i))
                pairs_count.append(mul/(k*k))
            
            max_idx = np.argmax(pairs_count)
            predicted_probab.append(pairs_count[max_idx])
            predicted_label.append(labelA[max_idx])
       
        return predicted_label, predicted_probab
        
        
if __name__ == '__main__':
    X = np.random.rand(10, 5)
    X_test = np.random.rand(5, 5)
    #X_dummy = np.random.rand(2, 5)
    Y = np.array([1,1,2,2,1,2,3,3,3,1])
    Y_test = np.array([1,1,3,1,2])
    print("Before MKNN transformation : ")
    print(X)
    mknn = MKNN()
    predicted_label, predicted_probab = mknn.fit(X, Y).marginalised_knn(X_test, k=3)
    print(predicted_label, predicted_probab)
    acc2 = np.sum(predicted_label == Y_test)/X_test.shape[0]    
    print('Accuracy after MKNN: ', acc2*100)