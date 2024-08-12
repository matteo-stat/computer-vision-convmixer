import keras
from typing import Tuple

def convmixer_block(layer: keras.Layer, name_prefix: str, filters: int, kernel_size: int | Tuple[int, int]) -> keras.Layer:
    """
    convmixer block implemented as described in the original paper "patches are all you need"
    basically it's a sequence of deptwhise convolution, a residual connection and pointwise convolution
    each convolution operation it's followed by a gelu activation and batch normalization


    Args:
        layer (keras.Layer): input layer
        name_prefix (str): prefix for give a proper name to each layer
        filters (int): number of filters to apply in the pointwise convolution
        kernel_size (int | Tuple[int, int]): kernel size to apply in the depthwise convolution

    Returns:
        keras.Layer: output from convmixer block
    """
    # save input layer for residual connection
    layer_input = layer

    # deptwhise separable convolution, followed by gelu activation and batch normalization
    layer = keras.layers.DepthwiseConv2D(kernel_size=kernel_size, padding='same', name=f'{name_prefix}-depthwise')(layer)
    layer = keras.layers.Activation('gelu', name=f'{name_prefix}-depthwise-gelu')(layer)
    layer = keras.layers.BatchNormalization(name=f'{name_prefix}-depthwise-batchnorm')(layer)

    # residual connection
    layer = keras.layers.Add(name=f'{name_prefix}-add')([layer, layer_input])

    # pointwise separable convolution, followed by gelu activation and batch normalization
    layer = keras.layers.Conv2D(filters=filters, kernel_size=1, name=f'{name_prefix}-pointwise')(layer)
    layer = keras.layers.Activation('gelu', name=f'{name_prefix}-pointwise-gelu')(layer)
    layer = keras.layers.BatchNormalization(name=f'{name_prefix}-pointwise-batchnorm')(layer)

    return layer
