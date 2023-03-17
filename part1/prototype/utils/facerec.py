import joblib
import numpy as np
from scipy.spatial.distance import mahalanobis
import tensorflow as tf


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
        return self.lda.transform(self.pca.transform(X))
    

## Face Recognition Classifiers

class MahalanobisDist:

    '''Loads the Mahalanobis classifier from the given path.'''
    '''The multiplier is used to determine the threshold for the Mahalanobis distance.'''

    def __init__(self, path):
        
        self._mu_fj = np.load(path)['mu_fj']
        self._VI = np.load(path)['inv_sigma_w_f']
        self._n_classes = self._mu_fj.shape[0]

    def predict(self, X, multiplier=1.5):

        if not len(X): return X
        y_preds, y_dist = [], []

        for i in range(len(X)):
            dist = np.array([mahalanobis(X[i], self._mu_fj[j], self._VI) for j in range(self._n_classes)])
            y_preds.append(np.argmin(dist))
            y_dist.append(dist)

        y_preds = np.array(y_preds)
        y_dist = np.array(y_dist)

        indices = (y_dist.mean(axis=1) - y_dist.min(axis=1)) > multiplier*y_dist.std(axis=1)
        y_preds[indices] = -1

        return y_preds

class KerasMLP:

    '''Loads the Keras MLP classifier from the given path.'''

    def __init__(self, path):

        self.model = tf.keras.models.load_model(path)

    def predict(self, X, score=0.5):

        if not len(X): return X

        y_preds = self.model.predict(X, verbose=0)
        indices = np.max(y_preds, axis=1) < score
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

    def predict(self, X, score=0.5):

        if not len(X): return X

        self.interpreter.resize_tensor_input(self.input_details[0]['index'], (len(X), self.n_feat_tflite))
        self.interpreter.resize_tensor_input(self.output_details[0]['index'], (len(X), 5))
        self.interpreter.allocate_tensors()

        self.interpreter.set_tensor(self.input_details[0]['index'], X)
        self.interpreter.invoke()
        y_preds = self.interpreter.get_tensor(self.output_details[0]['index'])

        indices = np.max(y_preds, axis=1) < score
        y_preds = np.argmax(y_preds, axis=1)
        y_preds[indices] = -1

        return y_preds