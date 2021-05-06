"""
Contains architecture definitions for discriminator models.

Author: Neel Dey
"""

import numpy as np
import tensorflow.keras.layers as KL
from tensorflow_addons.layers import SpectralNormalization as SN
from tensorflow.keras import Model

from .blocks import discblock, ResBlockD
from .layers import GroupMaxPool, GlobalSumPooling2D


def discriminator_model(img_shape, disc_arch='p4_food101', nclasses=6):
    """
    Return a tf image discriminator model.
    TODO: Redundant code, remember to clean up.

    Args:
        img_shape: tuple
            Input image shape.
        disc_arch: str
            Discriminator arch in format "x_y" where x in {"z2", "p4", "p4m"}
            and y in {"anhir", "lysto", "rotmnist", "cifar10", "food101"}
        nclasses: int
            Number of categories in the image set.
    """
    # Input images:
    input_img = KL.Input(shape=img_shape, name="main_input")

    if disc_arch == 'p4_anhir':
        sc = np.sqrt(8)

        # Define convolutional sequence:
        fea = ResBlockD(input_img, nfilters=int(32//sc), h_input='Z2',
                        h_output='D4', pad='same', stride=1)
        fea = ResBlockD(fea, nfilters=int(64//sc), h_input='D4',
                        h_output='D4', pad='same', stride=1)
        fea = ResBlockD(fea, nfilters=int(128//sc), h_input='D4',
                        h_output='D4', pad='same', stride=1)
        fea = ResBlockD(fea, nfilters=int(256//sc), h_input='D4',
                        h_output='D4', pad='same', stride=1)
        fea = ResBlockD(fea, nfilters=int(512//sc), h_input='D4',
                        h_output='D4', pad='same', stride=1)

        # Group pool:
        fea = KL.Activation('relu')(fea)
        fea = (GroupMaxPool('D4'))(fea)

        flat = KL.GlobalAveragePooling2D()(fea)

        # Input class label:
        label = KL.Input(shape=(nclasses,))
        label_emb = SN(KL.Dense(int(512//sc)))(label)

    elif disc_arch == 'z2_anhir':
        sc = np.sqrt(8)

        # Define convolutional sequence:
        fea = ResBlockD(input_img, nfilters=32, h_input='Z2',
                        h_output='Z2', pad='same', stride=1,
                        group_equiv=False)
        fea = ResBlockD(fea, nfilters=64, h_input='Z2',
                        h_output='Z2', pad='same', stride=1,
                        group_equiv=False)
        fea = ResBlockD(fea, nfilters=128, h_input='Z2',
                        h_output='Z2', pad='same', stride=1,
                        group_equiv=False)
        fea = ResBlockD(fea, nfilters=256, h_input='Z2',
                        h_output='Z2', pad='same', stride=1,
                        group_equiv=False)
        fea = ResBlockD(fea, nfilters=512, h_input='Z2',
                        h_output='Z2', pad='same', stride=1,
                        group_equiv=False)

        # Group pool:
        fea = KL.Activation('relu')(fea)
        flat = KL.GlobalAveragePooling2D()(fea)

        # Input class label:
        label = KL.Input(shape=(nclasses,))
        label_emb = SN(KL.Dense(512))(label)

    elif disc_arch == 'p4m_lysto':
        sc = np.sqrt(8)

        # Define convolutional sequence:
        fea = ResBlockD(input_img, nfilters=int(16//sc), h_input='Z2',
                        h_output='D4', pad='same', stride=1, BN=True)
        fea = ResBlockD(fea, nfilters=int(32//sc), h_input='D4',
                        h_output='D4', pad='same', stride=1, BN=True)
        fea = ResBlockD(fea, nfilters=int(64//sc), h_input='D4',
                        h_output='D4', pad='same', stride=1, BN=True)
        fea = ResBlockD(fea, nfilters=int(128//sc), h_input='D4',
                        h_output='D4', pad='same', stride=1, BN=True)
        fea = ResBlockD(fea, nfilters=int(256//sc), h_input='D4',
                        h_output='D4', pad='same', stride=1, BN=True)
        fea = ResBlockD(fea, nfilters=int(512//sc), h_input='D4',
                        h_output='D4', pad='same', stride=1, BN=True)

        # Group pool:
        fea = KL.Activation('relu')(fea)
        fea = (GroupMaxPool('D4'))(fea)

        flat = KL.GlobalAveragePooling2D()(fea)

        # Input class label:
        label = KL.Input(shape=(nclasses,))
        label_emb = SN(KL.Dense(int(512//sc)))(label)

    elif disc_arch == 'z2_lysto':
        sc = np.sqrt(8)

        # Define convolutional sequence:
        fea = ResBlockD(input_img, nfilters=16, h_input='Z2',
                        h_output='Z2', pad='same', stride=1,
                        group_equiv=False, BN=True)
        fea = ResBlockD(fea, nfilters=32, h_input='Z2',
                        h_output='Z2', pad='same', stride=1,
                        group_equiv=False, BN=True)
        fea = ResBlockD(fea, nfilters=64, h_input='Z2',
                        h_output='Z2', pad='same', stride=1,
                        group_equiv=False, BN=True)
        fea = ResBlockD(fea, nfilters=128, h_input='Z2',
                        h_output='Z2', pad='same', stride=1,
                        group_equiv=False, BN=True)
        fea = ResBlockD(fea, nfilters=256, h_input='Z2',
                        h_output='Z2', pad='same', stride=1,
                        group_equiv=False, BN=True)
        fea = ResBlockD(fea, nfilters=512, h_input='Z2',
                        h_output='Z2', pad='same', stride=1,
                        group_equiv=False, BN=True)

        # Group pool:
        fea = KL.Activation('relu')(fea)

        flat = KL.GlobalAveragePooling2D()(fea)

        # Input class label:
        label = KL.Input(shape=(nclasses,))
        label_emb = SN(KL.Dense(512))(label)

    elif disc_arch == 'p4_rotmnist':
        # Define convolutional sequence:
        fea = discblock(input_img, nfilters=64, h_input='Z2',
                        h_output='C4', pool='avg')
        fea = discblock(fea, nfilters=128, h_input='C4',
                        h_output='C4', pool='avg')
        fea = discblock(fea, nfilters=256, h_input='C4',
                        h_output='C4', pool='avg')
        # Group pool D4 filters into Z2:
        fea = (GroupMaxPool('C4'))(fea)

        flat = KL.GlobalAveragePooling2D()(fea)

        # Input class label:
        label = KL.Input(shape=(nclasses,))
        label_emb = KL.Dense(256)(label)

    elif disc_arch == 'z2_rotmnist':
        # Define convolutional sequence:
        fea = discblock(input_img, nfilters=128, h_input='Z2',
                        h_output='Z2', group_equiv=False, pool='avg')
        fea = discblock(fea, nfilters=256, h_input='Z2',
                        h_output='Z2', group_equiv=False, pool='avg')
        fea = discblock(fea, nfilters=512, h_input='Z2',
                        h_output='Z2', group_equiv=False, pool='avg')

        flat = KL.GlobalAveragePooling2D()(fea)

        # Input class label:
        label = KL.Input(shape=(nclasses,))
        label_emb = KL.Dense(512)(label)

    elif disc_arch == 'z2_food101':
        sc = np.sqrt(4)

        # Define convolutional sequence:
        fea = ResBlockD(input_img, nfilters=128, h_input='Z2',
                        h_output='Z2', group_equiv=False)
        fea = ResBlockD(fea, nfilters=256, h_input='Z2',
                        h_output='Z2', group_equiv=False)
        fea = ResBlockD(fea, nfilters=512, h_input='Z2',
                        h_output='Z2', group_equiv=False)
        fea = ResBlockD(fea, nfilters=784, h_input='Z2',
                        h_output='Z2', group_equiv=False)

        fea = KL.Activation('relu')(fea)

        flat = KL.GlobalAveragePooling2D()(fea)

        # Input class label:
        label = KL.Input(shape=(nclasses,))
        label_emb = SN(KL.Dense(784))(label)

    elif disc_arch == 'p4_food101':
        sc = np.sqrt(4)

        # Define convolutional sequence:
        fea = ResBlockD(input_img, nfilters=int(128//sc), h_input='Z2',
                        h_output='C4', pad='same', stride=1)
        fea = ResBlockD(fea, nfilters=int(256//sc), h_input='C4',
                        h_output='C4', pad='same', stride=1)
        fea = ResBlockD(fea, nfilters=int(512//sc), h_input='C4',
                        h_output='C4', pad='same', stride=1)
        fea = GroupMaxPool('C4')(fea)
        fea = ResBlockD(
            fea, nfilters=int(784), h_input='Z2', h_output='Z2',
            pad='same', stride=1, group_equiv=False,
        )

        fea = KL.Activation('relu')(fea)

        flat = KL.GlobalAveragePooling2D()(fea)

        # Input class label:
        label = KL.Input(shape=(nclasses,))
        label_emb = SN(KL.Dense(int(784)))(label)

    elif disc_arch == 'p4_cifar10':
        sc = np.sqrt(4)

        fea = ResBlockD(input_img, nfilters=int(128//sc), h_input='Z2',
                        h_output='C4', poolshort='avg', poolskip='avg')
        fea = ResBlockD(fea, nfilters=int(128//sc), h_input='C4',
                        h_output='C4', poolshort='avg', poolskip='avg')
        fea = ResBlockD(fea, nfilters=int(128//sc), h_input='C4',
                        downsample=False, h_output='C4',
                        poolshort='avg', poolskip='avg')
        fea = GroupMaxPool('C4')(fea)
        fea = ResBlockD(fea, nfilters=int(128), h_input='Z2', downsample=False,
                        h_output='Z2', group_equiv=False,
                        poolshort='avg', poolskip='avg')

        fea = KL.Activation('relu')(fea)

        flat = GlobalSumPooling2D()(fea)

        # Input class label:
        label = KL.Input(shape=(nclasses,))
        label_emb = SN(KL.Dense(int(128)))(label)

    elif disc_arch == 'z2_cifar10':

        # Define convolutional sequence:
        fea = ResBlockD(input_img, nfilters=128, h_input='Z2',
                        poolshort='avg', poolskip='avg', h_output='Z2',
                        group_equiv=False)
        fea = ResBlockD(fea, nfilters=128, h_input='Z2', poolshort='avg',
                        poolskip='avg', h_output='Z2', group_equiv=False)
        fea = ResBlockD(fea, nfilters=128, h_input='Z2', downsample=False,
                        h_output='Z2', group_equiv=False)
        fea = ResBlockD(fea, nfilters=128, h_input='Z2', downsample=False,
                        h_output='Z2', group_equiv=False)

        fea = KL.Activation('relu')(fea)

        flat = GlobalSumPooling2D()(fea)

        # Input class label:
        label = KL.Input(shape=(nclasses,))
        label_emb = SN(KL.Dense(128))(label)

    # Projection discriminator:
    projection = KL.dot([flat, label_emb], axes=1)
    op = SN(KL.Dense(1))(flat)

    prediction = KL.Add()([projection, op])
    model = Model(inputs=[input_img, label], outputs=prediction)

    return model
