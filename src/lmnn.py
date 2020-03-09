import numpy as np
from collections import Counter
from sklearn.metrics import pairwise_distances



class LMNN:
    def __init__(self, k=3, min_iter=50, max_iter=1000, learn_rate=1e-7,regularization=0.5, convergence_tol=0.001):
        self.k = k
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.learn_rate = learn_rate
        self.regularization = regularization
        self.convergence_tol = convergence_tol

    def process_inputs(self, X, labels):
        num_pts = X.shape[0]
        assert len(labels) == num_pts
        unique_labels, self.label_inds = np.unique(labels, return_inverse=True)
        self.labels = np.arange(len(unique_labels))
        self.X = X
        self.L = np.eye(X.shape[1])
        required_k = np.bincount(self.label_inds).min()
        k = self.k
        assert k <= required_k, ('not enough class labels for specified k' +
                                ' (smallest class has %d)' % required_k)


    def pairwise_L2(self, A, B):
        return ((A-B)**2).sum(axis=1)


    def metric(self):
        return self.L.T.dot(self.L)

    def transform(self, X=None):
        if X is None:
            X = self.X
        return self.L.dot(X.T).T

    def select_targets(self):
        k = self.k
        target_neighbors = np.empty((self.X.shape[0], k), dtype=int)
        for label in self.labels:
            inds, = np.nonzero(self.label_inds == label)
            dd = pairwise_distances(self.X[inds])
            np.fill_diagonal(dd, np.inf)
            nn = np.argsort(dd)[...,:k]
            target_neighbors[inds] = inds[nn]
        return target_neighbors

    def find_impostors(self, furthest_neighbors):
        Lx = self.transform()
        margin_radii = self.pairwise_L2(Lx, Lx[furthest_neighbors]) + 1
        impostors = []
        for label in self.labels[:-1]:
            in_inds, = np.nonzero(self.label_inds == label)
            out_inds, = np.nonzero(self.label_inds > label)
            dist = pairwise_distances(Lx[out_inds], Lx[in_inds])
            i1,j1 = np.nonzero(dist < margin_radii[out_inds][:,None])
            i2,j2 = np.nonzero(dist < margin_radii[in_inds])
            i = np.hstack((i1,i2))
            j = np.hstack((j1,j2))
            ind = np.vstack((i,j)).T
            if ind.size > 0:
                # gross: get unique rows
                ind = np.array(list(set(map(tuple,ind))))
            i,j = np.atleast_2d(ind).T
            impostors.append(np.vstack((in_inds[j], out_inds[i])))
        return np.hstack(impostors)

    def sum_outer_products(self, data, a_inds, b_inds, weights=None):
        Xab = data[a_inds] - data[b_inds]
        if weights is not None:
            return np.dot(Xab.T, Xab * weights[:,None])
        return np.dot(Xab.T, Xab)

    def count_edges(self, act1, act2, impostors, targets):
        imp = impostors[0,act1]
        c = Counter(zip(imp, targets[imp]))
        imp = impostors[1,act2]
        c.update(zip(imp, targets[imp]))
        if c:
            a = []
            for i in c:
                a.append(i)
            active_pairs = np.array(a)
        else:
            active_pairs = np.empty((0,2), dtype=int)
        v = []
        for i in c.values():
            v.append(i)
        f = np.array(v)
        return active_pairs, f


    def fit(self, X, Y):
        self.X = X
        self.labels = Y
        self.process_inputs(X, self.labels)
        target_neighbors = self.select_targets()
        impostors = self.find_impostors(target_neighbors[:,-1])
        dfG = self.sum_outer_products(self.X, target_neighbors.flatten(),np.repeat(np.arange(self.X.shape[0]), self.k))
        df = np.zeros_like(dfG)
        a1 = [None]*self.k
        a2 = [None]*self.k
        for nn_idx in range(self.k):
            a1[nn_idx] = np.array([])
            a2[nn_idx] = np.array([])

        # initialize gradient and L
        G = dfG * self.regularization + df * (1-self.regularization)
        L = self.L
        objective = np.inf
        for it in range(self.max_iter):
            df_old = df.copy()
            a1_old = [a.copy() for a in a1]
            a2_old = [a.copy() for a in a2]
            objective_old = objective
            Lx = L.dot(self.X.T).T
            g0 = self.pairwise_L2(*Lx[impostors])
            Ni = np.sum((Lx[:,None,:] - Lx[np.array(target_neighbors).astype('int')])**2, axis=2) + 1
            g1, g2 = Ni[impostors]


            total_active = 0
            for nn_idx in reversed(range(self.k)):
                act1 = g0 < g1[:,nn_idx]
                act2 = g0 < g2[:,nn_idx]
                total_active += act1.sum() + act2.sum()

                if it > 1:
                    plus1 = act1 & ~a1[nn_idx]
                    minus1 = a1[nn_idx] & ~act1
                    plus2 = act2 & ~a2[nn_idx]
                    minus2 = a2[nn_idx] & ~act2
                else:
                    plus1 = act1
                    plus2 = act2
                    minus1 = np.zeros(0, dtype=int)
                    minus2 = np.zeros(0, dtype=int)
                targets = target_neighbors[:,nn_idx]
                PLUS, pweight = self.count_edges(plus1, plus2, impostors, targets)
                # keylist = []
                # keylist.extend(iter(PLUS))
                # print(pweight)
                
                # print(self.sum_outer_products(self.X, PLUS[:,0], PLUS[:,1], pweight))
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
            assert not np.isnan(df).any()
            G = dfG * self.regularization + df * (1-self.regularization)
            objective = total_active * (1-self.regularization)
            objective += G.flatten().dot(L.T.dot(L).flatten())
            assert not np.isnan(objective)
            delta_obj = objective - objective_old
            if delta_obj > 0:
                # we're getting worse... roll back!
                self.learn_rate /= 2.0
                df = df_old
                a1 = a1_old
                a2 = a2_old
                objective = objective_old
            else:
                # update L
                L -= self.learn_rate * 2 * L.dot(G)
                self.learn_rate *= 1.01
            if it > self.min_iter and abs(delta_obj) < self.convergence_tol:
                break
        self.L = L
        return self


# if __name__ == '__main__':
#     X = np.random.rand(10, 5)
#     Y = np.array([1,1,2,2,1,2,3,3,3,1])
#     lmnn = LMNN()
#     X_transformed = lmnn.fit(X, Y).transform(X)
    