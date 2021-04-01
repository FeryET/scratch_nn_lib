from sklearn.metrics import accuracy_score
import numpy as np

class SoftmaxAccuracy:
    def __call__(self, y_true, y_pred, *args, **kwds):
        y_pred_maxed = np.zeros_like(y_true)
        y_pred_maxed[:, np.argmax(y_pred, axis=1)] = 1
        return accuracy_score(y_true=y_true, y_pred=y_pred_maxed)