import tensorflow as tf


def get_optimizers(
    lr_g, beta1_g, beta2_g, lr_d, beta1_d, beta2_d,
):
    """
    Function to return Adam optimizer objects. Note the calls to
    optimizer.iterations and optimizer.decay. They need to be called for TF2
    checkpointing for some reason.

    Args
        lr_g: float
            generator learning rate.
        beta1_g: float
            generator beta_1 Adam parameter.
        beta2_g: float
            generator beta_2 Adam parameter.
        lr_d: float
            discriminator learning rate.
        beta1_d: float
            discriminator beta_1 Adam parameter.
        beta2_d: float
            discriminator beta_2 Adam parameter.
    """
    generator_optimizer = tf.keras.optimizers.Adam(
        learning_rate=tf.Variable(lr_g),
        beta_1=tf.Variable(beta1_g),
        beta_2=tf.Variable(beta2_g),
        epsilon=tf.Variable(1e-7),
    )
    generator_optimizer.iterations
    generator_optimizer.decay = tf.Variable(0.0)

    discriminator_optimizer = tf.keras.optimizers.Adam(
        learning_rate=tf.Variable(lr_d),
        beta_1=tf.Variable(beta1_d),
        beta_2=tf.Variable(beta2_d),
        epsilon=tf.Variable(1e-7),
    )
    discriminator_optimizer.iterations
    discriminator_optimizer.decay = tf.Variable(0.0)

    return generator_optimizer, discriminator_optimizer
