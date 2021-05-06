from tensorflow.keras import backend as K
from tensorflow.keras import initializers as initializations
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.utils import get_custom_objects
from tensorflow.python.keras.layers.pooling import GlobalPooling2D

from groupy.gconv.tensorflow_gconv.splitgconv2d import gconv2d_util


# -----------------------------------------------------------------------------
# Misc layers:

class GlobalSumPooling2D(GlobalPooling2D):
    """2D global sum pooling layer. Not in default tf, for some reason."""
    def call(self, inputs):
        if self.data_format == 'channels_last':
            return K.sum(inputs, axis=[1, 2])
        else:
            return K.sum(inputs, axis=[2, 3])

# -----------------------------------------------------------------------------
# Conditional scaling and shifting layers:


class GFiLM(Layer):
    """
    Group-equivariant conditional affine transformations for 2D feature maps.
    Modified from:
    stackoverflow.com/questions/55210684/feature-wise-scaling-and-shifting-film-layer-in-keras
    Class design inspired by the implementation of Group Batch Normalization.
    """

    def __init__(self, h, axis=-1, **kwargs):
        if axis != -1:
            raise ValueError(
                'Assumes 2D input with channels as last dimension.',
            )

        self.h = h
        self.axis = axis

        super(GFiLM, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        feature_map_shape, FiLM_gamma_shape, FiLM_beta_shape = input_shape

        def divider(w):
            number = w
            if self.h == 'C4':
                number //= 4
            elif self.h == 'D4':
                number //= 8
            elif self.h == 'Z2':
                number //= 1
            else:
                raise ValueError('Wrong h: %s' % self.h)
            return number

        self.n_feature_maps = divider(feature_map_shape[-1])

        self.height = feature_map_shape[1]
        self.width = feature_map_shape[2]

        assert(int(self.n_feature_maps) == FiLM_gamma_shape[1])
        assert(int(self.n_feature_maps) == FiLM_beta_shape[1])

        super(GFiLM, self).build(input_shape)

    def call(self, x):
        assert isinstance(x, list)
        conv_output, FiLM_gamma, FiLM_beta = x

        FiLM_gamma = K.expand_dims(FiLM_gamma, axis=[1])
        FiLM_gamma = K.expand_dims(FiLM_gamma, axis=[1])
        FiLM_gamma = K.tile(FiLM_gamma, [1, self.height, self.width, 1])

        FiLM_beta = K.expand_dims(FiLM_beta, axis=[1])
        FiLM_beta = K.expand_dims(FiLM_beta, axis=[1])
        FiLM_beta = K.tile(FiLM_beta, [1, self.height, self.width, 1])

        def repeat(w):
            n = 1
            if self.h == 'C4':
                n *= 4
            elif self.h == 'D4':
                n *= 8
            elif self.h == 'Z2':
                n *= 1
            else:
                raise ValueError('Wrong h: %s' % self.h)

            return K.reshape(
                K.tile(K.expand_dims(w, -1), [1, 1, 1, 1, n]),
                [-1, self.height, self.width, self.n_feature_maps*n],
            )

        repeated_gamma = repeat(FiLM_gamma)
        repeated_beta = repeat(FiLM_beta)

        # Apply affine transformation
        return (1 + repeated_gamma) * conv_output + repeated_beta

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return input_shape[0]

    def get_config(self):
        return dict(list({'h': self.h}.items()) +
                    list(super(GFiLM, self).get_config().items()))


class FiLM(Layer):
    """
    Conditional affine transformations for 2D feature maps.
    Modified from:
    stackoverflow.com/questions/55210684/feature-wise-scaling-and-shifting-film-layer-in-keras

    TODO: merge with GFiLM, separate classes are redundant here.
    """

    def __init__(self, **kwargs):
        super(FiLM, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        feature_map_shape, FiLM_gamma_shape, FiLM_beta_shape = input_shape

        self.height = feature_map_shape[1]
        self.width = feature_map_shape[2]
        self.n_feature_maps = feature_map_shape[-1]

        assert(int(self.n_feature_maps) == FiLM_gamma_shape[1])
        assert(int(self.n_feature_maps) == FiLM_beta_shape[1])

        super(FiLM, self).build(input_shape)

    def call(self, x):
        assert isinstance(x, list)
        conv_output, FiLM_gamma, FiLM_beta = x

        FiLM_gamma = K.expand_dims(FiLM_gamma, axis=[1])
        FiLM_gamma = K.expand_dims(FiLM_gamma, axis=[1])
        FiLM_gamma = K.tile(FiLM_gamma, [1, self.height, self.width, 1])

        FiLM_beta = K.expand_dims(FiLM_beta, axis=[1])
        FiLM_beta = K.expand_dims(FiLM_beta, axis=[1])
        FiLM_beta = K.tile(FiLM_beta, [1, self.height, self.width, 1])

        # Apply affine transformation
        return (1 + FiLM_gamma) * conv_output + FiLM_beta

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return input_shape[0]


# -----------------------------------------------------------------------------
# Misc. G-equivariant layers covering missing layers in keras-gcnn
# TODO: merge with keras-gcnn

class GroupMaxPool(Layer):
    """Max pool over orientations."""
    def __init__(self, h_input, **kwargs):
        super(GroupMaxPool, self).__init__(**kwargs)
        self.h_input = h_input

    def build(self, input_shape):
        self.shape = input_shape
        super(GroupMaxPool, self).build(input_shape)

    @property
    def nti(self):
        nti = 1
        if self.h_input == 'C4':
            nti *= 4
        elif self.h_input == 'D4':
            nti *= 8
        return nti

    def call(self, x):
        shape = K.int_shape(x)

        input_reshaped = K.reshape(
            x,
            (-1, shape[1], shape[2], shape[3] // self.nti, self.nti),
        )
        max_per_group = K.max(input_reshaped, -1)

        return max_per_group

    def compute_output_shape(self, input_shape):
        return (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3] // self.nti,
        )

    def get_config(self):
        config = super(GroupMaxPool, self).get_config()
        config['h_input'] = self.h_input
        return config


class GShift(Layer):
    """
    Convolutional bias terms are not currently supported by keras-gcnn, so this
    layer fills that gap, until I update keras-gcnn to support biases.

    Class design inspired by keras-gcnn Group Batch Normalization.
    """

    def __init__(self, h, axis=-1, beta_initializer='zeros', **kwargs):
        self.h = h
        if axis != -1:
            raise ValueError(
                'Assumes 2D input with channels as last dimension.',
            )

        self.axis = axis
        self.beta_initializer = initializations.get(beta_initializer)

        if self.h == 'C4':
            self.n = 4
        elif self.h == 'D4':
            self.n = 8
        elif self.h == 'Z2':
            self.n = 1
        else:
            raise ValueError('Wrong h: %s' % self.h)

        super(GShift, self).__init__(**kwargs)

    def build(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError(
                'Axis ' + str(self.axis) + ' of '
                'input tensor should have a defined dimension '
                'but the layer received an input with shape ' +
                str(input_shape) + '.',
            )
        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})

        self.gconv_indices, self.gconv_shape_info, w_shape = gconv2d_util(
            h_input=self.h, h_output=self.h,
            in_channels=input_shape[-1],
            out_channels=input_shape[-1],
            ksize=1,
        )
        if self.h == 'C4':
            dim //= 4
        elif self.h == 'D4':
            dim //= 8
        shape = (dim,)

        self.beta = self.add_weight(shape=shape,
                                    name='beta',
                                    initializer=self.beta_initializer)

        self.broadcast_shape = [1] * len(input_shape)
        self.broadcast_shape[self.axis] = input_shape[self.axis]

        self.built = True

    def call(self, inputs, training=None):

        repeated_beta = K.reshape(
                K.tile(
                    K.expand_dims(self.beta, -1), [1, self.n]), [-1])
        out = inputs + K.reshape(repeated_beta, self.broadcast_shape)

        return out

    def get_config(self):
        return dict(list({'h': self.h}.items()) +
                    list(super(GShift, self).get_config().items()))


get_custom_objects().update({'GFiLM': GFiLM})
get_custom_objects().update({'FiLM': FiLM})
get_custom_objects().update({'GroupMaxPool': GroupMaxPool})
get_custom_objects().update({'GShift': GShift})
