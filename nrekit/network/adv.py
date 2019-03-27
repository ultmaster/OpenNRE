import tensorflow as tf

def adversarial(loss, embedding):
    perturb = tf.gradients(loss, embedding)
    perturb = tf.reshape((0.01 * tf.stop_gradient(tf.nn.l2_normalize(perturb, dim=[0, 1, 2]))), [-1, FLAGS.max_length, embedding.shape[-1]])
    embedding = embedding + perturb
    return embedding
