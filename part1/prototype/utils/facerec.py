import joblib
import numpy as np
from scipy.spatial.distance import mahalanobis
import tensorflow as tf


class PCA:

    '''Loads the PCA model from the given path.'''

    def __init__(self, path):
        
        self.pca = joblib.load(path)
        # self._VI = np.diag(1 / self.pca.explained_variance_[:n_components])

    def transform(self, X):

        if X is None: return None
        return self.pca.transform(X)
    
    # def get_VI(self):

    #     return self._VI
    

class MahalanobisClassifier:

    '''Loads the Mahalanobis classifier from the given path.'''
    '''The multiplier is used to determine the threshold for the Mahalanobis distance.'''

    def __init__(self, path):
        
        self._mu_fj = np.load(path)['mu_fj']
        self._n_classes = self._mu_fj.shape[0]
        self._VI = np.load(path)['inv_sigma_w_f']

    def predict(self, X, multiplier=1.5):

        if X is None: return np.empty((0,), dtype=int)

        y_preds, y_dist = [], []

        for i in range(len(X)):
            dist = np.array([mahalanobis(X[i], self._mu_fj[j], self._VI) for j in range(self._n_classes)])
            y_preds.append(np.argmin(dist))
            y_dist.append(dist)

        y_preds = np.array(y_preds)
        y_dist = np.array(y_dist)

        # indices = (y_dist.mean(axis=1) - y_dist.min(axis=1)) > multiplier*y_dist.std(axis=1)
        # y_preds[indices] = -1

        return y_preds
    

class KerasClassifier:

    '''Loads the Keras classifier from the given path.'''

    def __init__(self, path):

        self.model = tf.keras.models.load_model(path)

    def predict(self, X, score=0.5):

        if X is None: return np.empty((0,), dtype=int)

        y_preds = self.model.predict(X, verbose=0)
        indices = np.max(y_preds, axis=1) < score
        y_preds = np.argmax(y_preds, axis=1)
        y_preds[indices] = -1

        return y_preds
