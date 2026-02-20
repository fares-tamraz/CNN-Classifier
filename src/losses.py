"""Canonical FocalLoss for all scripts in this project.

Import convention:
  - From src/ scripts (run as `python src/script.py`):
      from losses import FocalLoss
  - From repo-root scripts:
      from src.losses import FocalLoss
"""

import tensorflow as tf


class FocalLoss(tf.keras.losses.Loss):
    """Focal Loss with optional label smoothing.

    Works with sparse (integer) labels and any number of classes.
    label_smoothing=0.0 disables smoothing (default, backward-compatible).
    """

    def __init__(self, alpha=1.0, gamma=2.0, label_smoothing=0.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        num_classes = tf.shape(y_pred)[-1]
        y_true_onehot = tf.one_hot(tf.cast(y_true, tf.int32), num_classes)
        y_true_onehot = tf.cast(y_true_onehot, y_pred.dtype)

        if self.label_smoothing > 0:
            y_true_onehot = (
                y_true_onehot * (1 - self.label_smoothing)
                + self.label_smoothing / tf.cast(num_classes, y_pred.dtype)
            )

        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        ce = -y_true_onehot * tf.math.log(y_pred)
        ce = tf.reduce_sum(ce, axis=-1)

        p_t = tf.reduce_sum(y_true_onehot * y_pred, axis=-1)
        focal_weight = tf.pow(1.0 - p_t, self.gamma)

        return tf.reduce_mean(self.alpha * focal_weight * ce)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "alpha": self.alpha,
                "gamma": self.gamma,
                "label_smoothing": self.label_smoothing,
            }
        )
        return config
