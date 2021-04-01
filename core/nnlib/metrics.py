from sklearn.metrics import accuracy_score
import numpy as np

class SoftmaxAccuracy:
    def __call__(self, y_true, y_pred, *args, **kwds):
        y_maxed = np.argmax(y_true, axis=1)
        y_pred_maxed = np.argmax(y_pred, axis=1)
        return (y_maxed == y_pred_maxed).astype(np.int32).mean()