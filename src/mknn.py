import numpy as np
import os
import cv2
import glob
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt



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
    print('Data preprocessing done...')
    print('Data Shape:', X.shape)
    print('label Shape:', Y.shape)

    # TSNE plot before LMNN transformstion.
    plot_tsne(X, Y, 'TSNE_before')
    print('TSNE plot saved')
    
    # Normalize the dataset
    ### TODO

    # Reduce dimensionality if needed
    ### TODO

    # LMNN Transform of data is Euclidean space.
    ### TODO
    
    