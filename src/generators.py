"""
Contains architecture definitions for generator models.

Author: Neel Dey
"""

import tensorflow.keras.layers as KL
from tensorflow.keras.models import Model
from tensorflow_addons.layers import SpectralNormalization as SN

from keras_gcnn.layers import GConv2D, GBatchNorm

from .utils.data_utils import generator_dimensionality
from .layers import GShift, GroupMaxPool
from .blocks import CCBN, ResBlockG_film


def generator_model(nclasses=101, gen_arch='p4_food101', latent_dim=64):
    """
    Return a tf image generator model.

    Args:
        nclasses: int
            Number of categories in the image set.
        gen_arch: str
            Generator arch in format "x_y" where x in {"z2", "p4", "p4m"}
            and y in {"anhir", "lysto", "rotmnist", "cifar10", "food101"}
        latent_dim: int
            Dimensionality of Gaussian latents.
    """

    # Get latent vector:
    latent_vec = KL.Input(shape=(latent_dim,))
    label_vec = KL.Input(shape=(nclasses,))

    sc, proj_dim, proj_shape, labelemb_dim = generator_dimensionality(gen_arch)
    label_proj = KL.Dense(  # Using SN here seems to lead to collapse
        labelemb_dim,
        use_bias=False,
    )(label_vec)

    # Concatenate noise and condition feature maps to modulate the generator:
    cla = KL.concatenate([latent_vec, label_proj])

    # Project and reshape to spatial feature maps:
    gen = SN(KL.Dense(proj_dim))(cla)
    gen = KL.Reshape(proj_shape)(gen)

    if gen_arch == 'p4m_anhir':
        # (Residual) Convolutions + Upsampling:
        fea = ResBlockG_film(
            gen, cla, int(512//sc), h_input='Z2', h_output='D4', pad='same',
        )
        for ch in [256, 128, 64, 32]:
            fea = ResBlockG_film(
                fea, cla, int(ch//sc), h_input='D4', h_output='D4', pad='same',
            )

        fea = GBatchNorm(h='D4', momentum=0.1)(fea)
        fea = KL.Activation('relu')(fea)

        fea = SN(GConv2D(3, kernel_size=3, h_input='D4', h_output='D4',
                         padding='same', kernel_initializer='orthogonal'))(fea)
        op = GroupMaxPool('D4')(fea)

    elif gen_arch == 'z2_anhir':
        # (Residual) Convolutions + Upsampling:
        fea = ResBlockG_film(
            gen, cla, 512, h_input='Z2', h_output='Z2',
            pad='same', stride=1, group_equiv=False,
        )
        for ch in [256, 128, 64, 32]:
            fea = ResBlockG_film(
                fea, cla, ch, h_input='Z2', h_output='Z2',
                pad='same', group_equiv=False,
            )

        fea = KL.BatchNormalization(
            momentum=0.1, center=False, scale=False,
        )(fea)
        fea = CCBN(fea, cla, 'Z2')
        fea = KL.Activation('relu')(fea)

        op = SN(KL.Conv2D(
            3, kernel_size=3, padding='same', use_bias=False,
        ))(fea)

    elif gen_arch == 'p4m_lysto':
        # (Residual) Convolutions + Upsampling:
        fea = ResBlockG_film(
            gen, cla, int(512//sc), h_input='Z2', h_output='D4', pad='same',
        )
        for ch in [256, 128, 64, 32, 16]:
            fea = ResBlockG_film(
                fea, cla, int(ch//sc), h_input='D4', h_output='D4', pad='same',
            )

        fea = GBatchNorm(h='D4', momentum=0.1)(fea)
        fea = KL.Activation('relu')(fea)

        fea = SN(GConv2D(3, kernel_size=3, h_input='D4', h_output='D4',
                         padding='same', kernel_initializer='orthogonal'))(fea)
        op = GroupMaxPool('D4')(fea)

    elif gen_arch == 'z2_lysto':
        # (Residual) Convolutions + Upsampling:
        fea = ResBlockG_film(
            gen, cla, 512, h_input='Z2', h_output='Z2',
            pad='same', group_equiv=False,
        )
        for ch in [256, 128, 64, 32, 16]:
            fea = ResBlockG_film(
                fea, cla, ch, h_input='Z2', h_output='Z2',
                pad='same', group_equiv=False,
            )

        fea = KL.BatchNormalization(momentum=0.1)(fea)
        fea = KL.Activation('relu')(fea)

        op = SN(KL.Conv2D(
            3, kernel_size=3, padding='same', use_bias=False,
        ))(fea)

    elif gen_arch == 'p4_rotmnist':
        # Convolutions + Upsampling:
        fea = SN(GConv2D(256, kernel_size=3, h_input='Z2',
                         h_output='C4', padding='same'))(gen)
        fea = GShift(h='C4')(fea)
        fea = KL.Activation('relu')(fea)

        fea = KL.UpSampling2D()(fea)
        fea = SN(GConv2D(128, kernel_size=3, h_input='C4',
                         h_output='C4', padding='same'))(fea)
        fea = GBatchNorm(h='C4', momentum=0.1, center=False, scale=False)(fea)
        fea = CCBN(fea, cla, 'C4')
        fea = KL.Activation('relu')(fea)

        fea = KL.UpSampling2D()(fea)
        fea = SN(GConv2D(64, kernel_size=3, h_input='C4',
                         h_output='C4', padding='same'))(fea)
        fea = GBatchNorm(h='C4', momentum=0.1, center=False, scale=False)(fea)
        fea = CCBN(fea, cla, 'C4')
        fea = KL.Activation('relu')(fea)

        fea = SN(GConv2D(1, kernel_size=3, h_input='C4',
                         h_output='C4', padding='same'))(fea)
        op = GroupMaxPool('C4')(fea)

    elif gen_arch == 'z2_rotmnist':
        # Convolutions + Upsampling:
        fea = SN(KL.Conv2D(512, kernel_size=3, padding='same',
                           use_bias=True,
                           kernel_initializer='orthogonal'))(gen)
        fea = KL.Activation('relu')(fea)

        fea = KL.UpSampling2D()(fea)
        fea = SN(KL.Conv2D(256, kernel_size=3, padding='same',
                           use_bias=False,
                           kernel_initializer='orthogonal'))(fea)
        fea = KL.BatchNormalization(
            momentum=0.1, center=False, scale=False,
        )(fea)
        fea = CCBN(fea, cla, 'Z2')
        fea = KL.Activation('relu')(fea)

        fea = KL.UpSampling2D()(fea)
        fea = SN(KL.Conv2D(128, kernel_size=3, padding='same',
                           use_bias=False,
                           kernel_initializer='orthogonal'))(fea)
        fea = KL.BatchNormalization(
            momentum=0.1, center=False, scale=False,
        )(fea)
        fea = CCBN(fea, cla, 'Z2')
        fea = KL.Activation('relu')(fea)

        op = SN(KL.Conv2D(1, kernel_size=3, padding='same', use_bias=False,
                          kernel_initializer='orthogonal'))(fea)

    elif gen_arch == 'p4_food101':
        # (Residual) Convolutions + Upsampling:
        fea = ResBlockG_film(
            gen, cla, int(512//sc), h_input='Z2', h_output='C4', pad='same',
        )
        for ch in [384, 256, 192]:
            fea = ResBlockG_film(
                fea, cla, int(ch//sc), h_input='C4', h_output='C4', pad='same',
            )

        fea = GBatchNorm(h='C4', momentum=0.1)(fea)
        fea = KL.Activation('relu')(fea)

        fea = SN(GConv2D(3, kernel_size=3, h_input='C4', h_output='C4',
                         padding='same', kernel_initializer='orthogonal'))(fea)
        op = GroupMaxPool('C4')(fea)

    elif gen_arch == 'z2_food101':
        # (Residual) Convolutions + Upsampling:
        fea = ResBlockG_film(gen, cla, 512, h_input='Z2',
                             h_output='Z2', pad='same', group_equiv=False)
        fea = ResBlockG_film(fea, cla, 384, h_input='Z2',
                             h_output='Z2', pad='same', group_equiv=False)
        fea = ResBlockG_film(fea, cla, 256, h_input='Z2',
                             h_output='Z2', pad='same', group_equiv=False)
        fea = ResBlockG_film(fea, cla, 192, h_input='Z2',
                             h_output='Z2', pad='same', group_equiv=False)

        fea = KL.BatchNormalization(momentum=0.1)(fea)
        fea = KL.Activation('relu')(fea)

        op = SN(KL.Conv2D(3, kernel_size=3, padding='same',
                          kernel_initializer='orthogonal'))(fea)

    elif gen_arch == 'p4_cifar10':
        # (Residual) Convolutions + Upsampling:
        fea = ResBlockG_film(
            gen, cla, int(256//sc), h_input='Z2', h_output='C4', pad='same',
        )
        fea = ResBlockG_film(
            fea, cla, int(256//sc), h_input='C4', h_output='C4', pad='same',
        )
        fea = ResBlockG_film(
            fea, cla, int(256//sc), h_input='C4', h_output='C4', pad='same',
        )

        fea = GBatchNorm(h='C4', momentum=0.1)(fea)
        fea = KL.Activation('relu')(fea)

        fea = SN(GConv2D(3, kernel_size=3, h_input='C4', h_output='C4',
                         padding='same', kernel_initializer='orthogonal'))(fea)
        op = GroupMaxPool('C4')(fea)

    elif gen_arch == 'z2_cifar10':
        # (Residual) Convolutions + Upsampling:
        fea = ResBlockG_film(
            gen, cla, 256, h_input='Z2', h_output='Z2',
            pad='same', group_equiv=False,
        )
        fea = ResBlockG_film(
            fea, cla, 256, h_input='Z2', h_output='Z2',
            pad='same', stride=1, group_equiv=False,
        )
        fea = ResBlockG_film(
            fea, cla, 256, h_input='Z2', h_output='Z2',
            pad='same', stride=1, group_equiv=False,
        )

        fea = KL.BatchNormalization(momentum=0.1)(fea)
        fea = KL.Activation('relu')(fea)

        op = SN(KL.Conv2D(
            3, kernel_size=3, padding='same', use_bias=False,
        ))(fea)

    else:
        raise ValueError('Generator Architecture Unrecognized')

    gen_img = KL.Activation('tanh')(op)  # Get final synthesized image batch

    # Generator model:
    generator = Model([latent_vec, label_vec], gen_img)

    return generator
