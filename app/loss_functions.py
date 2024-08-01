import tensorflow as tf
from tensorflow.keras.losses import Loss

class AsymmetricLoss(Loss):
    """
    For an engine degradation scenario an early prediction is
    preferred over late predictions. Therefore, the scoring
    algorithm was asymmetric around the truetime of failure 
    such that late predictions were more heavily penalized 
    than early predictions. In either case, the penalty
    grows exponentially with increasing error. The asymmetric
    preference is controlled by parameters a1 and a2.
    """
    def __init__(self, a1=10, a2=13, name="asymmetric_loss"):
        super().__init__(name=name)
        self.a1 = a1
        self.a2 = a2

    def call(self, y_true, y_pred):
        d = y_pred - y_true
        loss = tf.where(d < 0,
                        tf.exp(-d / self.a1) - 1,
                        tf.exp(d / self.a2) - 1)
        tf.print("d:", d)
        tf.print("loss:", loss)
        return tf.reduce_mean(loss)
