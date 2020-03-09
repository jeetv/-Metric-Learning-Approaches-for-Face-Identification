import numpy as np
import os
import cv2
import glob
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from lmnn import LMNN
from sklearn.neighbors import KNeighborsClassifier



def preprocess(dir):
    '''
        This method preprocesses the dataset and returns two array X and Y
        X: contains the original data
        Y: contains the labels

        input:
            dir -> Directory of the dataset.
    '''
    dirlist = os.listdir(dir)
    X = []
    Y = []
    for path in tqdm(dirlist):
        file = glob.glob(dir +'/'+ path + '/*.jpg')
        for i in file:
            img = cv2.imread(i)
            img = img.reshape((250, 250, 3))
            X.append(img.reshape(-1))
            Y.append(path)
    X = np.array(X)
    Y = np.array(Y)
    uniquey = enumerate(np.unique(Y))
    uniquey = {val: i for i, val in uniquey}
    y = []
    for i in Y:
        y.append(uniquey[i])
    Y = np.array(y)
    return X, Y


def select_labels(X_train, Y_train, k=3):
    '''
        This method discards lables with less than k no of samples
        input:
            X_train -> data
            Y_train -> labels
            k -> min no of samples required per label(default 3)
        returns:
            X, Y
    '''
    classes = []
    for i in np.unique(Y_train):
        if np.sum(Y_train == i) < k:
            classes.append(i)

    classes = np.array(classes)
    X_new = []
    Y_new = []
    for i in range(X_train.shape[0]):
        if Y_train[i] not in classes:
            X_new.append(X_train[i])
            Y_new.append(Y_train[i])
    X_new = np.array(X_new)
    Y_new = np.array(Y_new)
    return X_new, Y_new


def plot_tsne(X, Y, name='TSNE'):
    '''
        this method makes TSNE plots.
    '''
    # TSNE plot of first 2000 samples with 500 features.
    X = X[:2000, :500]
    Y = Y[:2000]
    tsne = TSNE(n_components=2, random_state=0).fit_transform(X)
    plt.scatter(tsne[:,0], tsne[:,1], c=Y)
    plt.title('TSNE Plot')
    if not os.path.isdir('../Plots'):
        os.mkdir('../Plots')
    plt.savefig('../Plots/'+name+'.jpg')
    plt.close()


if __name__ == '__main__':
    dir = '../dataset'

    print('Data Preprocessing...')
    X, Y = preprocess(dir)
    X = X[:1000]
    Y = Y[:1000]
    print('Data preprocessing done...')
    print('Data Shape:', X.shape)
    print('label Shape:', Y.shape)
    # Normalize the dataset
    mean = np.mean(X)
    var = np.var(X)
    X = (X - mean)/var
    print('data normalized')

    # Reduce dimensionality if needed
    pca = PCA(n_components=200)
    X_pca = pca.fit(X).transform(X)
    print('dimension reduced..')
    print('data shape:', X_pca.shape)
    print('label shape:', Y.shape)
    
    # train test split
    X_train, X_test, Y_train, Y_test = train_test_split(X_pca, Y, test_size=0.2, random_state=42)
    print(X_train.shape)


    X_new, Y_new = select_labels(X_train, Y_train, 5)
    print(X_new.shape, Y_new.shape)
    X_new_test, Y_new_test = select_labels(X_test, Y_test, 5)
    print('Performing KNN')
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_new, Y_new)
    pred1 = neigh.predict(X_new_test)
    acc1 = np.sum(pred1 == Y_new_test)/X_new_test.shape[0]
    print('Accuracy before LMNN: ', acc1*100)
    # TSNE plot before LMNN transformstion.
    # plot_tsne(X_new, Y_new, 'TSNE_before')
    # print('TSNE plot saved')


    # LMNN Transform of data from Euclidean space.
    lmnn = LMNN()
    lmnn = lmnn.fit(X_new, Y_new)
    X_transformed = lmnn.transform(X_new)
    X_test_transformed = lmnn.transform(X_new_test)

    print('Performing KNN')
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_transformed, Y_new)
    pred2 = neigh.predict(X_test_transformed)
    acc2 = np.sum(pred2 == Y_new_test)/X_test_transformed.shape[0]
    print('Accuracy after LMNN: ', acc2*100)
    # TSNE plot after LMNN transformstion.
    # plot_tsne(X_new, Y_new, 'TSNE_after')
    # print('TSNE plot saved')
    
    
    
