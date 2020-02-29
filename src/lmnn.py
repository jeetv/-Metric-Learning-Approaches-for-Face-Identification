import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy import sparse, optimize
from sklearn.utils.extmath import row_norms, safe_sparse_dot



class LMNN:
    def __init__(self, n_neighbours=2, n_features_out=None, max_itr=200):
        self.n_neighbours = n_neighbours
        self.n_features_out = n_features_out
        self.max_itr = max_itr
    

    def select_target_neighbour(self):
        '''
            This method finds out target neighbours of each sample.
            returns:
                array called neighbours of shape (X.shape[0], n_neighbours)
        '''
        neighbours = np.zeros((self.X.shape[0], self.n_neighbours))
        for classes in self._classes:
            idx, = np.where(np.equal(self.Y, classes))
            dist = euclidean_distances(self.X[idx], squared=True)
            np.fill_diagonal(dist, np.inf)
            neigh_ind = np.argpartition(dist, self.n_neighbours - 1, axis=1)
            neigh_ind = neigh_ind[:, :self.n_neighbours]
            row_ind = np.arange(len(idx))[:, None]
            neigh_ind = neigh_ind[row_ind, np.argsort(dist[row_ind, neigh_ind])]
            neighbours[idx] = idx[neigh_ind]
        
        return neighbours


    def compute_grad_static(self):
        '''
            This function computes gradient component of the target neighbours that will not 
            change through out training. This gradient is used to pull same class samples closer.
            returns:
                An array with the sum of all outer products of
                (sample, target_neighbor) pairs.
        '''
        n_samples, n_neighbors = self.target_neighbours.shape
        row = np.repeat(range(n_samples), n_neighbors)
        col = self.target_neighbours.ravel()
        tn_graph = sparse.csr_matrix((np.ones(self.target_neighbours.size), (row, col)),
                              shape=(n_samples, n_samples)).todense()
        weights_sym = tn_graph + tn_graph.T
        # print(weights_sym)
        # print(weights_sym.sum(1).getA())
        # print(safe_sparse_dot(weights_sym, self.X, dense_output=True))
        diagonal = weights_sym.sum(1).getA()
        laplacian_dot_X = diagonal * self.X - safe_sparse_dot(weights_sym, self.X,
                                                        dense_output=True)
        result = np.dot(self.X.T, laplacian_dot_X)
        return result


    def init_transform(self):
        L = np.eye(self.X.shape[1])
        self.n_features_out = L.shape[0] if self.n_features_out is None else self.n_features_out
        self.n_features_in = self.X.shape[1]
        if self.n_features_out > self.n_features_in:
            self.n_features_out = self.n_features_in
        return L, self.n_features_out

    def fit(self, X, Y):
        #initializing data
        self.X = X
        #initializing labels
        self.Y = Y
        self._classes = np.unique(Y)

        # storing target neighbours of each samples(will be same through out training)
        self.target_neighbours = self.select_target_neighbour()
        self.grad_static = self.compute_grad_static()

        L, self.n_features_out = self.init_transform()
        # print(L.shape, self.X.shape, self.n_features_out)
        
        
        


if __name__ == '__main__':
    x = np.array([[0,0],[-1,0.1],[0.3,-0.05],[0.7,0.3],[-0.2,-0.6],[-0.15,-0.63],[-0.25,0.55],[-0.28,0.67]])
    # x = np.random.rand(10, 5)
    y = np.array([0,0,0,0,1,1,2,2])
    lmnn = LMNN()
    lmnn.fit(x, y)
