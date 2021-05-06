"""
Main training script for Group Equivariant GANs, ICLR 2021.

Author: Neel Dey
"""

import datetime
import os
import random
import numpy as np
import tensorflow as tf
import tensorflow.math as tfm

from time import time
from tensorflow.compat.v1 import set_random_seed
from tensorflow.keras.utils import Progbar

from src.discriminators import discriminator_model
from src.generators import generator_model
from src.optimizers import get_optimizers
from src.utils.data_utils import dataset_lookup, npy_loader, data_generator
from src.utils.training_args import training_args

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Parse CLI:
args = training_args()

num_classes, print_multiplier = dataset_lookup(args.dataset)

# Set RNG for numpy and tensorflow
np.random.seed(args.rng)
set_random_seed(args.rng)
random.seed(args.rng)

# Set format for directory names to save models in:
save_folder = (('{}_{}_loss_{}_{}eps_Garch_{}_Darch_{}'
                '_dupdates{}_lrg{}_lrd{}_gp{}_batchsize_{}')
               .format(
                   args.name, args.dataset, args.loss, args.epochs,
                   args.g_arch, args.d_arch, args.d_updates, args.lr_g,
                   args.lr_d, args.gp_wt, args.batchsize,
))

# ---------------------------------------------------------------------------
# Data loading

# Load dataset:
data, labels = npy_loader(args.dataset, num_classes)

# Set up data generator:
datagen = data_generator(
    data, labels, args.batchsize, args.latent_dim, args.dataset,
)

# ---------------------------------------------------------------------------
# Intialize networks

# Define generator and discriminator networks:
generator = generator_model(
    nclasses=num_classes, gen_arch=args.g_arch, latent_dim=args.latent_dim,
)

discriminator = discriminator_model(
    img_shape=data.shape[1:], nclasses=num_classes, disc_arch=args.d_arch,
)

# Create optimizers:
goptim, doptim = get_optimizers(
    args.lr_g, args.beta1_g, args.beta2_g,  # generator adam params
    args.lr_d, args.beta1_d, args.beta2_d,  # discriminator adam params
)


# ---------------------------------------------------------------------------
# Plotting and checkpointing setup:

# Setup training checkpoints:
checkpoint_dir = './training_checkpoints/{}'.format(save_folder)
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(
    generator_optimizer=goptim,
    discriminator_optimizer=doptim,
    generator=generator,
    discriminator=discriminator,
)

# Location to save training samples:
os.makedirs('training_pngs/' + save_folder, exist_ok=True)

# Parameters for samples visualized and saved as PNGs:
test_labels = np.transpose(
    np.tile(np.eye(num_classes), num_classes*print_multiplier),
)

# If resuming training
if args.resume_ckpt > 0:
    checkpoint.restore(
        './training_checkpoints/{}/ckpt-{}'.format(
            save_folder, args.resume_ckpt,
        )
    ).assert_consumed()

summary_writer = tf.summary.create_file_writer(
    'training_logs/{}'.format(save_folder) +
    datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
)


# ---------------------------------------------------------------------------
# Training loops

EPS = tf.convert_to_tensor(1e-6, dtype=tf.float32)


@tf.function
def gen_train_step(
    real_batch, label, noise_batch, step, eps=EPS,
):
    """
    Generator training step. Only supports the relativistic average loss for
    now.
    # TODO: abstract out loss and support more types of losses.

    Args:
        real_batch: np.array (batch_size, x, y, ch)
            Batch of randomly sampled real images.
        label: (batch_size, n_classes)
            Batch of labels corresponding to the randomly sampled reals above.
        noise_batch: np.array (batch_size, latent_dim)
            Batch of random latents.
        eps: tf float
            Constant to keep the log function happy.
    """

    with tf.GradientTape() as gen_tape:
        # Generate fake images to feed discriminator:
        fake_batch = generator([noise_batch, label], training=True)

        # Get discriminator logits on real images and fake images:
        # Could use different labels for fakes too. Doesn't make a
        # noticeable difference.
        disc_opinion_real = discriminator([real_batch, label], training=True)
        disc_opinion_fake = discriminator([fake_batch, label], training=True)

        # Get output for relativistic average losses:
        real_fake_rel_avg_opinion = (
            disc_opinion_real - tf.reduce_mean(disc_opinion_fake, axis=0)
        )

        fake_real_rel_avg_opinion = (
            disc_opinion_fake - tf.reduce_mean(disc_opinion_real, axis=0)
        )

        # Total generator loss:
        gen_loss = tf.reduce_mean(
            - tf.reduce_mean(
                tfm.log(tfm.sigmoid(fake_real_rel_avg_opinion) + eps),
                axis=0,
            )
            - tf.reduce_mean(
                tfm.log(1 - tfm.sigmoid(real_fake_rel_avg_opinion) + eps),
                axis=0,
            )
        )

    # Get gradients and update generator:
    generator_gradients = gen_tape.gradient(
        gen_loss, generator.trainable_variables,
    )

    goptim.apply_gradients(
        zip(generator_gradients, generator.trainable_variables),
    )

    with summary_writer.as_default():
        tf.summary.scalar('losses/g_loss', gen_loss, step=step)
        tf.summary.image('samples', 0.5*(fake_batch + 1), step=step)


@tf.function
def disc_train_step(
    real_batch, label, noise_batch, step, eps=EPS,
):
    """
    Discriminator training step. So far only supports the relavg_gp loss.
    # TODO: abstract out loss and support more types of losses.

    Args:
        real_batch: np.array (batch_size, x, y, ch)
            Batch of randomly sampled real images.
        label: (batch_size, n_classes)
            Batch of labels corresponding to the randomly sampled reals above.
        noise_batch: np.array (batch_size, latent_dim)
            Batch of random latents.
        eps: tf float
            Constant to keep the log function happy.
    """
    gp_strength = tf.constant(args.gp_wt, dtype=tf.float32)

    with tf.GradientTape() as disc_tape:
        # Generate fake images to feed discriminator:
        fake_batch = generator([noise_batch, label], training=True)

        # Get discriminator logits on real images and fake images:
        # Could use different labels for fakes too. Doesn't make a
        # noticeable difference.
        disc_opinion_real = discriminator([real_batch, label], training=True)
        disc_opinion_fake = discriminator([fake_batch, label], training=True)

        # Get output for relativistic average losses:
        real_fake_rel_avg_opinion = (
            disc_opinion_real - tf.reduce_mean(disc_opinion_fake, axis=0)
        )

        fake_real_rel_avg_opinion = (
            disc_opinion_fake - tf.reduce_mean(disc_opinion_real, axis=0)
        )

        # Get loss:
        disc_loss = tf.reduce_mean(
            - tf.reduce_mean(
                tfm.log(
                    tfm.sigmoid(real_fake_rel_avg_opinion) + eps), axis=0,
            )
            - tf.reduce_mean(
                tfm.log(
                    1 - tfm.sigmoid(fake_real_rel_avg_opinion) + eps), axis=0,
            )
        )

        # Get gradient penalty:
        new_real_batch = 1.0 * real_batch
        new_label = 1.0 * label
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(new_real_batch)
            disc_opinion_real_new = discriminator(
                [new_real_batch, new_label], training=True,
            )

        grad = gp_tape.gradient(disc_opinion_real_new, new_real_batch)
        grad_sqr = tfm.square(grad)
        grad_sqr_sum = tf.reduce_sum(
                grad_sqr,
                axis=np.arange(1, len(grad_sqr.shape)),
        )
        gradient_penalty = (gp_strength/2.0) * tf.reduce_mean(grad_sqr_sum)
        total_disc_loss = disc_loss + gradient_penalty

    # Get gradients and update discriminator:
    discriminator_gradients = disc_tape.gradient(
        total_disc_loss,
        discriminator.trainable_variables,
    )

    doptim.apply_gradients(
        zip(discriminator_gradients, discriminator.trainable_variables),
    )

    with summary_writer.as_default():
        tf.summary.scalar('losses/d_loss', disc_loss, step=step)
        tf.summary.scalar('regularizers/GP', gradient_penalty, step=step)


# ---------------------------------------------------------------------------
# Train loop:


for epoch in range(args.epochs):
    print("epoch {} of {}".format(epoch + 1, args.epochs))
    nbatches = args.batchsize * (args.d_updates + 1)

    # Print progress bar:
    progress_bar = Progbar(target=int(data.shape[0] // nbatches))

    # Loop through each batch:
    start_time = time()
    steps = int(data.shape[0] // nbatches)
    for index in range(steps):  # Loop through steps
        progress_bar.update(index)

        # Update discriminator:
        for j in range(args.d_updates):
            noise, image_batch, labs_batch = next(iter(datagen))

            disc_train_step(
                tf.convert_to_tensor(image_batch, dtype=tf.float32),
                tf.convert_to_tensor(labs_batch, dtype=tf.float32),
                tf.convert_to_tensor(noise, dtype=tf.float32),
                tf.convert_to_tensor((index + epoch*steps), dtype=tf.int64),
            )

        # Update Generator:
        noise, image_batch, labs_batch = next(iter(datagen))

        gen_train_step(
            tf.convert_to_tensor(image_batch, dtype=tf.float32),
            tf.convert_to_tensor(labs_batch, dtype=tf.float32),
            tf.convert_to_tensor(noise, dtype=tf.float32),
            tf.convert_to_tensor((index + epoch*steps), dtype=tf.int64),
        )

    print('\nTime required for epoch: {}'.format(time() - start_time))

    # Generate samples for visualization and save them:
    generator.trainable = False

    test_noise = np.random.randn(
        num_classes * num_classes * print_multiplier, args.latent_dim,
    )
    samples = generator.predict([test_noise, test_labels])

    generator.trainable = True
    samples = (samples + 1) / 2.0

    if args.dataset == 'food101':
        n_display = 20
    else:
        n_display = num_classes*print_multiplier

    # For aligning rows with categories:
    for i in range(n_display):
        newrows = np.reshape(
            samples[i * num_classes: i * num_classes + num_classes],
            (data.shape[2] * num_classes, data.shape[2], data.shape[-1]),
        )
        if i == 0:
            rows = newrows
        else:
            rows = np.concatenate((rows, newrows), axis=1)
    rows = np.squeeze(rows)

    plt.imsave('training_pngs/{}/epoch_{:04}.png'
               .format(save_folder, epoch), rows)

    if (epoch + 1) % args.ckpt_freq == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)
