import tensorflow as tf

tf.compat.v1.disable_v2_behavior()


def safe_div(numerator, denominator, name="value"):
  """Computes a safe divide which returns 0 if the denominator is zero.
  Note that the function contains an additional conditional check that is
  necessary for avoiding situations where the loss is zero causing NaNs to
  creep into the gradient computation.
  Args:
    numerator: An arbitrary `Tensor`.
    denominator: `Tensor` whose shape matches `numerator` and whose values are
      assumed to be non-negative.
    name: An optional name for the returned op.
  Returns:
    The element-wise value of the numerator divided by the denominator.
  """
  return tf.compat.v1.where(
      tf.compat.v1.greater(denominator, 0),
      tf.compat.v1.div(numerator, tf.where(
          tf.compat.v1.equal(denominator, 0),
          tf.compat.v1.ones_like(denominator), denominator)),
      tf.compat.v1.zeros_like(numerator),
      name=name)


def safe_log(x):
  """Computes a safe logarithm which returns 0 if x is zero."""
  return tf.compat.v1.where(
      tf.compat.v1.equal(x, 0),
      tf.compat.v1.zeros_like(x),
      tf.compat.v1.log(tf.compat.v1.maximum(1e-12, x)))

