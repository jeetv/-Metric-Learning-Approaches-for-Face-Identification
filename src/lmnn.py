import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import pairwise_distances


class LMNN:
    def __init__(self,k=2, min_iter=50, max_iter=1000, learn_rate=1e-7, regularization=0.5, convergence_tol=0.001):
        self.k = k
        self.min_itr = min_iter
        self.max_iter = max_iter
        self.learn_rate = learn_rate
        self.regularization = regularization
        self.convergence_tol = convergence_tol
    

    def transform(self, X=None):
        if X is None:
            X = self.X
        return self.L.dot(X.T).T


    def target_neighbours(self):
        target_neighbors = np.zeros((self.X.shape[0], self.k))
        for label in self.classes:
            inds, = np.nonzero(self.Y == label)
            dd = pairwise_distances(self.X[inds])
            np.fill_diagonal(dd, np.inf)
            nn = np.argsort(dd)[...,:self.k]
            target_neighbors[inds] = inds[nn]
        return target_neighbors
        
    
    def pairwise_L2(self, x, y):
        return np.sum((x-y)**2, axis=1)


    def find_impostors(self, furthest_index):
        Lx = self.transform()
        margin_radii = self.pairwise_L2(Lx, Lx[np.array(furthest_index).astype('int')]) + 1
        impostors = []
        for labels in self.classes[:-1]:
            in_inds, = np.nonzero(self.Y == labels)
            out_inds, = np.nonzero(self.Y > labels)
            # print(in_inds, out_inds)
            dist = pairwise_distances(Lx[out_inds], Lx[in_inds])
            # print(dist)
            i1,j1 = np.nonzero(dist < margin_radii[out_inds][:,None])
            i2,j2 = np.nonzero(dist < margin_radii[in_inds])
            # print(i2, j2)
            i = np.hstack((i1,i2))
            j = np.hstack((j1,j2))
            ind = np.vstack((i,j)).T
            if ind.size > 0:
                # gross: get unique rows
                ind = np.array(list(set(map(tuple,ind))))
            i,j = np.atleast_2d(ind).T
            impostors.append(np.vstack((in_inds[j], out_inds[i])))
        return np.hstack(impostors)

    def sum_outer_products(self, X, a_idx, b_idx, weights=None):
        Xab = X[np.array(a_idx).astype('int')] - X[np.array(b_idx).astype('int')]
        if weights is not None:
            return np.dot(Xab.T, Xab*weights[:,None])
        return np.dot(Xab.T, Xab)



    def fit(self, x, y):
        self.X = x
        self.Y = y
        self.classes = np.unique(self.Y)
        self.L = np.eye(self.X.shape[1])
        target_neighbours = self.target_neighbours()
        impostors = self.find_impostors(target_neighbours[:, -1])
        # print(target_neighbours)
        # print(impostors)
        dfg = self.sum_outer_products(self.X, target_neighbours.flatten(), np.repeat(np.arange(self.X.shape[0]), self.k))
        df = np.zeros_like(dfg)

        a1 = [None]*self.k
        a2 = [None]*self.k
        
        for idx in range(self.k):
            a1[idx] = np.array([])
            a2[idx] = np.array([])
        
        G = dfg*self.regularization + df*(1 - self.regularization)
        L = self.L
        objective = np.inf

        for i in range(self.max_iter):
            df_old = df.copy()
            a1_old = [a.copy() for a in a1]
            a1_old = [a.copy() for a in a2]
            objective_old = objective
            Lx = L.dot(self.X.T).T
            g0 = self.pairwise_L2(*Lx[impostors])
            Ni = np.sum((Lx[:,None,:] - Lx[np.array(target_neighbours).astype('int')])**2, axis=2) + 1
            g1, g2 = Ni[impostors]
            
            total_active = 0
            for nn_idx in reversed(range(self.k)):
                act1 = g0 < g1[:,nn_idx]
                act2 = g0 < g1[:,nn_idx]
                total_active = act1.sum() + act2.sum()


                if i > 1:
                    plus1 = act1 & ~a1[nn_idx]
                    minus1 = a1[nn_idx] & ~act1
                    plus2 = act2 & ~a2[nn_idx]
                    minus2 = a2[nn_idx] & ~act2
                else:
                    plus1 = act1
                    plus2 = act2
                    minus1 = np.zeros(0, dtype=int)
                    minus2 = np.zeros(0, dtype=int)
                
                targets = target_neighbours[:, nn_idx]
                PLUS, pweight = self.count_edges(plus1, plus2, impostors, targets)
                df += self.sum_outer_products(self.X, PLUS[:,0], PLUS[:,1], pweight)

                MINUS, mweight = self.count_edges(minus1, minus2, impostors, targets)
                df -= self.sum_outer_products(self.X, MINUS[:,0], MINUS[:,1], mweight)

                in_imp, out_imp = impostors
                df += self.sum_outer_products(self.X, in_imp[minus1], out_imp[minus1])
                df += self.sum_outer_products(self.X, in_imp[minus2], out_imp[minus2])

                df -= self.sum_outer_products(self.X, in_imp[plus1], out_imp[plus1])
                df -= self.sum_outer_products(self.X, in_imp[plus2], out_imp[plus2])


                a1[nn_idx] = act1
                a2[nn_idx] = act2

                G = dfG * self.regularization + df * (1-self.regularization)
                objective = total_active * (1-self.regularization)
                objective += G.flatten().dot(L.T.dot(L).flatten())

                delta_obj = objective - objective_old


        






if __name__ == '__main__':
    x = np.array([[0,0],[-1,0.1],[0.3,-0.05],[0.7,0.3],[-0.2,-0.6],[-0.15,-0.63],[-0.25,0.55],[-0.28,0.67]])
    y = np.array([0,0,0,0,1,1,2,2])

    lmnn = LMNN()
    lmnn.fit(x, y)

    # plotting TSNE before applying LMNN
    # tsne = TSNE(n_components=2, random_state=0).fit_transform(x)
    # plt.scatter(x[:, 0], x[:, 1], c=y)
    # plt.savefig('TSNE.jpg')
    # plt.close()



