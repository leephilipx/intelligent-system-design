import joblib
import numpy as np
from scipy.stats import chi2
import tensorflow as tf

def squared_mahalanobis_dist(X, mean, inv_cov):
    '''X and mean are arrays of shape (n_features, 1)'''
    return (X - mean) @ inv_cov @ (X - mean).T


## Dimensionality reduction models

class PCA:

    '''Loads the PCA model from the given path.'''

    def __init__(self, path):
        
        self.pca = joblib.load(path)

    def transform(self, X):

        if not len(X): return X
        return self.pca.transform(X)
    
class FisherFace:

    '''Loads the PCA + LDA model from the given path.'''    

    def __init__(self, path_pca, path_lda):

        self.pca = joblib.load(path_pca)
        self.lda = joblib.load(path_lda)

    def transform(self, X):

        if not len(X): return X
        return self.lda.transform(self.pca.transform(X)).astype(np.float32)
    

## Face Recognition Classifiers

class MahalanobisDistance:

    '''Loads the Mahalanobis distance classifier from the given path.'''
    '''The confidence is used to determine the threshold using a Chi-Square distribution.'''

    def __init__(self, path):
        
        self._mu_fj = np.load(path)['mu_fj']
        self._VI = np.load(path)['inv_sigma_w_f']
        self._n_classes = self._mu_fj.shape[0]
        self._n_features = self._mu_fj.shape[1]

    def predict(self, X, conf=0.9999):

        if not len(X): return X
        y_preds = []
        threshold = chi2.ppf(conf, self._n_features)

        for i in range(len(X)):
            dist = np.array([squared_mahalanobis_dist(X[i], self._mu_fj[j], self._VI) for j in range(self._n_classes)])
            min_dist_index = np.argmin(dist)
            y_preds.append(min_dist_index if dist[min_dist_index] < threshold else -1)

        return np.array(y_preds)

class KerasMLP:

    '''Loads the Keras MLP classifier from the given path.'''

    def __init__(self, path):

        self.model = tf.keras.models.load_model(path)

    def predict(self, X, conf=0.5):

        if not len(X): return X

        y_preds = self.model.predict(X, verbose=0)
        indices = np.max(y_preds, axis=1) < conf
        y_preds = np.argmax(y_preds, axis=1)
        y_preds[indices] = -1

        return y_preds
    
class TfLiteMLP:

    '''Loads the TensorFlow Lite MLP classifier from the given path.'''

    def __init__(self, path):

        self.interpreter = tf.lite.Interpreter(model_path=path)
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.n_feat_tflite = self.input_details[0]['shape'][1]

    def predict(self, X, conf=0.5):

        if not len(X): return X

        self.interpreter.resize_tensor_input(self.input_details[0]['index'], (len(X), self.n_feat_tflite))
        self.interpreter.resize_tensor_input(self.output_details[0]['index'], (len(X), 5))
        self.interpreter.allocate_tensors()

        self.interpreter.set_tensor(self.input_details[0]['index'], X)
        self.interpreter.invoke()
        y_preds = self.interpreter.get_tensor(self.output_details[0]['index'])

        indices = np.max(y_preds, axis=1) < conf
        y_preds = np.argmax(y_preds, axis=1)
        y_preds[indices] = -1

        return y_preds


class LogisticRegression:

    def __init__(self, path):

        self.model = joblib.load(path)

    def predict(self, X, conf=0.5):

        if not len(X): return X

        y_preds = self.model.predict_proba(X)
        indices = np.max(y_preds, axis=1) < conf
        y_preds = np.argmax(y_preds, axis=1)
        y_preds[indices] = -1

        return y_preds
    

class SupportVectorMachine:

    def __init__(self, path):

        self.model = joblib.load(path)

    def predict(self, X, conf=0.5):

        if not len(X): return X
        return self.model.predict(X)