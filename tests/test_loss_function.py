import unittest
import tensorflow as tf
import numpy as np
from app.loss_functions import AsymmetricLoss

class TestAsymmetricLoss(unittest.TestCase):

    def setUp(self):
        self.loss_fn = AsymmetricLoss(a1=10, a2=13)

    def test_loss_positive_diff(self):
        y_true = tf.constant([10.0])
        y_pred = tf.constant([15.0])
        expected_loss = np.exp(5.0 / 13.0) - 1  # since d > 0, use a2
        loss = self.loss_fn(y_true, y_pred)
        self.assertAlmostEqual(loss.numpy(), expected_loss, places=5)

    def test_loss_negative_diff(self):
        y_true = tf.constant([15.0])
        y_pred = tf.constant([10.0])
        expected_loss = np.exp(5.0 / 10.0) - 1  # since d < 0, use a1
        loss = self.loss_fn(y_true, y_pred)
        self.assertAlmostEqual(loss.numpy(), expected_loss, places=5)

    def test_loss_zero_diff(self):
        y_true = tf.constant([10.0])
        y_pred = tf.constant([10.0])
        expected_loss = 0.0
        loss = self.loss_fn(y_true, y_pred)
        self.assertAlmostEqual(loss.numpy(), expected_loss, places=5)

    def test_loss_multiple_values(self):
        y_true = tf.constant([10.0, 15.0, 20.0])
        y_pred = tf.constant([15.0, 10.0, 20.0])
        expected_loss = (np.exp(5.0 / 13.0) - 1 + np.exp(5.0 / 10.0) - 1 + 0.0) / 3
        loss = self.loss_fn(y_true, y_pred)
        self.assertAlmostEqual(loss.numpy(), expected_loss, places=5)

if __name__ == '__main__':
    unittest.main()
