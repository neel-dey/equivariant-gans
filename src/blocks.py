"""
"""
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL
from tensorflow_addons.layers import SpectralNormalization as SN

from keras_gcnn.layers import GConv2D, GBatchNorm

from .layers import GShift, GFiLM, FiLM


# -----------------------------------------------------------------------------
# Generator Blocks

# add optional arg for kernel init
def CCBN(feat, cla, group, specnorm=True, initialization='orthogonal'):
    """
    Class-conditional batch normalization.
    TODO: Support more than just orthogonal kernel initializers.

    Args:
        feat: tf Tensor
            Input feature map to modulate.
        cla: tf Tensor
            Input vector (here latent+condition) to synthesize images with.
        group: str
            Symmetry group to be equivariant to. One of {'Z2', 'C4', 'D4'}.
        specnorm: bool
            Whether to use spectral normalization on the linear projections.
        initialization: str
            Kernel initializer for linear projection.
    """
    channels = K.int_shape(feat)[-1]

    if group == 'D4':
        channels = int(channels // 8)
    elif group == 'C4':
        channels = int(channels // 4)
    elif group == 'Z2':
        channels = int(channels // 1)
    else:
        raise ValueError('Unknown Group')

    if specnorm is True:
        x_beta = SN(KL.Dense(
            channels, kernel_initializer=initialization,
        ))(cla)
        x_gamma = SN(KL.Dense(
            channels, kernel_initializer=initialization,
        ))(cla)
    else:
        x_beta = KL.Dense(channels, kernel_initializer=initialization)(cla)
        x_gamma = KL.Dense(channels, kernel_initializer=initialization)(cla)

    if group != 'Z2':
        return GFiLM(h=group)([feat, x_gamma, x_beta])
    else:
        return FiLM()([feat, x_gamma, x_beta])


def ResBlockG_film(
    feat, cla, nfilters, h_input, h_output, pad, stride=1,
    group_equiv=True, kernel_size=3, bn_eps=1e-3, upsample=True,
):  # TODO: add optional arg for kernel init
    """
    A conditionally-modulated upsampling residual block for the Generator.

    Args:
        feat: tf Tensor
            Input feature map to modulate.
        cla: tf Tensor
            Input vector (here latent+condition) to synthesize images with.
        nfilters: int
            Number of convolutional filters.
        h_input: str
            Input group. One of {'Z2', 'C4', 'D4'}.
        h_output: str
            Output group. One of {'Z2', 'C4', 'D4'}.
        pad: str
            Zero-padding mode. One of {'valid', 'same'}.
        stride: int
            Convolutional stride.
        group_equiv: bool
            Whether to be equivariant.
        kernel_size: int
            Convolutional kernel size.
        bn_eps: float
            Numerical constant to keep the BatchNorm denominator happy.
        upsample: bool
            Whether to nearest-neighbors upsample 2x.
    """

    if group_equiv is True:
        # Shortcut connection:
        if upsample is True:
            shortcut = KL.UpSampling2D()(feat)
            shortcut = SN(GConv2D(nfilters,
                                  kernel_size=1,
                                  h_input=h_input,
                                  h_output=h_output,
                                  strides=stride,
                                  padding=pad,
                                  kernel_initializer='orthogonal'))(shortcut)
        elif upsample is False:
            shortcut = SN(GConv2D(nfilters,
                                  kernel_size=1,
                                  h_input=h_input,
                                  h_output=h_output,
                                  strides=stride,
                                  padding=pad,
                                  kernel_initializer='orthogonal'))(feat)

        # Convolutional path:
        if h_input == 'Z2':
            skip = KL.BatchNormalization(momentum=0.1, epsilon=bn_eps,
                                         center=False,
                                         scale=False)(feat)
            skip = CCBN(skip, cla, h_input)
        else:
            skip = GBatchNorm(h=h_input, epsilon=bn_eps,
                              momentum=0.1,
                              center=False,
                              scale=False)(feat)
            skip = CCBN(skip, cla, h_input)

        skip = KL.Activation('relu')(skip)
        if upsample is True:
            skip = KL.UpSampling2D()(skip)
        skip = SN(GConv2D(nfilters,
                          kernel_size=kernel_size,
                          h_input=h_input,
                          h_output=h_output,
                          strides=stride,
                          padding=pad,
                          kernel_initializer='orthogonal'))(skip)
        skip = GBatchNorm(h=h_output, momentum=0.1,
                          center=False, scale=False, epsilon=bn_eps)(skip)
        skip = CCBN(skip, cla, h_output)
        skip = KL.Activation('relu')(skip)
        skip = SN(GConv2D(nfilters,
                          kernel_size=kernel_size,
                          h_input=h_output,
                          h_output=h_output,
                          strides=stride,
                          padding=pad,
                          kernel_initializer='orthogonal'))(skip)

        # Add outputs:
        out = KL.Add()([shortcut, skip])

    elif group_equiv is False:
        # Shortcut connection:
        if upsample is True:
            shortcut = KL.UpSampling2D()(feat)
            shortcut = SN(KL.Conv2D(nfilters,
                                    kernel_size=1,
                                    strides=stride,
                                    padding=pad,
                                    kernel_initializer='orthogonal',
                                    use_bias=False))(shortcut)
        else:
            shortcut = SN(KL.Conv2D(nfilters,
                                    kernel_size=1,
                                    strides=stride,
                                    padding=pad,
                                    kernel_initializer='orthogonal',
                                    use_bias=False))(feat)

        # Convolutional path:
        skip = KL.BatchNormalization(momentum=0.1,
                                     center=False, scale=False, epsilon=bn_eps,
                                     )(feat)
        skip = CCBN(skip, cla, 'Z2')
        skip = KL.Activation('relu')(skip)
        if upsample is True:
            skip = KL.UpSampling2D()(skip)
        skip = SN(KL.Conv2D(nfilters,
                            kernel_size=3,
                            strides=stride,
                            padding=pad,
                            kernel_initializer='orthogonal',
                            use_bias=False))(skip)
        skip = KL.BatchNormalization(momentum=0.1,
                                     center=False, scale=False, epsilon=bn_eps,
                                     )(skip)
        skip = CCBN(skip, cla, 'Z2')
        skip = KL.Activation('relu')(skip)
        skip = SN(KL.Conv2D(nfilters,
                            kernel_size=3,
                            strides=stride,
                            padding=pad,
                            kernel_initializer='orthogonal',
                            use_bias=False))(skip)

        # Add outputs:
        out = KL.Add()([shortcut, skip])

    return out


# -----------------------------------------------------------------------------
# Discriminator blocks

def discblock(
    feat, nfilters, h_input, h_output, group_equiv=True,
    kernel_size=3, downsample=True, pool='max',
):
    """
    A Conv/SpecNorm/Activation block for the discriminator architectures.

    Args:
        feat: tf Tensor
            Input feature map to modulate.
        nfilters: int
            Number of convolutional filters.
        h_input: str
            Input group. One of {'Z2', 'C4', 'D4'}.
        h_output: str
            Output group. One of {'Z2', 'C4', 'D4'}.
        group_equiv: bool
            Whether to be equivariant.
        kernel_size: int
            Convolutional kernel size.
        downsample: bool
            Whether to downsample 2x.
        pool: str
            Pooling mode. One of {'avg', 'max'}.
    """
    if group_equiv is True:
        feat = SN(GConv2D(nfilters,
                          kernel_size=kernel_size,
                          h_input=h_input,
                          h_output=h_output,
                          padding='same',
                          kernel_initializer='orthogonal'))(feat)
        feat = GShift(h=h_output)(feat)
        feat = KL.LeakyReLU(alpha=0.2)(feat)
        if downsample is True:
            if pool == 'max':
                feat = KL.MaxPooling2D()(feat)
            elif pool == 'avg':
                feat = KL.AveragePooling2D()(feat)
    else:
        feat = SN(KL.Conv2D(nfilters,
                            kernel_size=kernel_size,
                            padding='same',
                            use_bias=True,
                            kernel_initializer='orthogonal'))(feat)
        feat = KL.LeakyReLU(alpha=0.2)(feat)
        if downsample is True:
            if pool == 'max':
                feat = KL.MaxPooling2D()(feat)
            elif pool == 'avg':
                feat = KL.AveragePooling2D()(feat)
    return feat


def ResBlockD(
    feat, nfilters, h_input, h_output, pad='same', stride=1, BN=False,
    group_equiv=True, kernel_size=3, downsample=True,
    poolshort='max', poolskip='max',
):
    """
    A residual block for the discriminator with spectral and/or batch norm.

    Args:
        feat: tf Tensor
            Input feature map to modulate.
        nfilters: int
            Number of convolutional filters.
        h_input: str
            Input group. One of {'Z2', 'C4', 'D4'}.
        h_output: str
            Output group. One of {'Z2', 'C4', 'D4'}.
        pad: str
            Zero-padding mode. One of {'valid', 'same'}.
        stride: int
            Convolutional stride.
        BN: bool
            Whether to use BatchNorm.
        group_equiv: bool
            Whether to be equivariant.
        kernel_size: int
            Convolutional kernel size.
        downsample: bool
            Whether to downsample 2x.
        poolshort: str
            Pooling mode. One of {'avg', 'max'}.
        poolskip: str
            Pooling mode. One of {'avg', 'max'}.
    """
    if group_equiv is True:
        # Shortcut block:
        shortcut = SN(GConv2D(nfilters,
                              kernel_size=1,
                              h_input=h_input,
                              h_output=h_output,
                              strides=stride,
                              padding=pad,
                              kernel_initializer='orthogonal'))(feat)
        shortcut = GShift(h=h_output)(shortcut)
        if downsample is True:
            if poolshort == 'max':
                shortcut = KL.MaxPooling2D((2, 2))(shortcut)
            elif poolshort == 'avg':
                shortcut = KL.AveragePooling2D((2, 2))(shortcut)

        # Skip connection:
        if BN is True:
            if h_input == 'Z2':
                skip = KL.BatchNormalization(momentum=0.1)(feat)
            else:
                skip = GBatchNorm(h=h_input, momentum=0.1)(feat)
            skip = KL.Activation('relu')(skip)
        else:
            skip = KL.Activation('relu')(feat)

        skip = SN(GConv2D(nfilters,
                          kernel_size=kernel_size,
                          h_input=h_input,
                          h_output=h_output,
                          strides=stride,
                          padding=pad,
                          kernel_initializer='orthogonal'))(skip)

        if BN is True:
            skip = GBatchNorm(h=h_output, momentum=0.1)(skip)
        else:
            skip = GShift(h=h_output)(skip)

        skip = KL.Activation('relu')(skip)
        skip = SN(GConv2D(nfilters,
                          kernel_size=kernel_size,
                          h_input=h_output,
                          h_output=h_output,
                          strides=stride,
                          padding=pad,
                          kernel_initializer='orthogonal'))(skip)
        skip = GShift(h=h_output)(skip)   # paper model had no shift here

        if downsample is True:
            if poolskip == 'max':
                skip = KL.MaxPooling2D((2, 2))(skip)
            elif poolskip == 'avg':
                skip = KL.AveragePooling2D((2, 2))(skip)

        # Residual addition:
        out = KL.Add()([shortcut, skip])

    elif group_equiv is False:
        # Shortcut:
        shortcut = SN(KL.Conv2D(nfilters,
                                kernel_size=1,
                                strides=stride,
                                padding=pad,
                                use_bias=True,
                                kernel_initializer='orthogonal'))(feat)
        if downsample is True:
            if poolshort == 'max':
                shortcut = KL.MaxPooling2D((2, 2))(shortcut)
            elif poolshort == 'avg':
                shortcut = KL.AveragePooling2D((2, 2))(shortcut)

        # Skip connection:
        if BN is True:
            skip = KL.BatchNormalization(momentum=0.1)(feat)
            skip = KL.Activation('relu')(skip)
            skip = SN(KL.Conv2D(nfilters,
                                kernel_size=kernel_size,
                                strides=stride,
                                padding=pad,
                                use_bias=False,
                                kernel_initializer='orthogonal'))(skip)
            skip = KL.BatchNormalization(momentum=0.1)(skip)
        else:
            skip = KL.Activation('relu')(feat)
            skip = SN(KL.Conv2D(nfilters,
                                kernel_size=kernel_size,
                                strides=stride,
                                padding=pad,
                                use_bias=True,
                                kernel_initializer='orthogonal'))(skip)

        skip = KL.Activation('relu')(skip)
        skip = SN(KL.Conv2D(nfilters,
                            kernel_size=kernel_size,
                            strides=stride,
                            padding=pad,
                            use_bias=True,  # paper model had this set to False
                            kernel_initializer='orthogonal'))(skip)
        if downsample is True:
            if poolskip == 'max':
                skip = KL.MaxPooling2D((2, 2))(skip)
            elif poolskip == 'avg':
                skip = KL.AveragePooling2D((2, 2))(skip)

        # Residual addition:
        out = KL.Add()([shortcut, skip])

    return out
