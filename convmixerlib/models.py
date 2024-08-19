from typing import Tuple,  Optional
import keras
from convmixerlib import blocks

def build_convmixer_classifier(
        input_shape: Tuple[int, int, int],
        patches_embedding_dimension: int,
        depth: int,
        patch_size: int,
        kernel_size: int | Tuple[int, int],
        num_classes: int,
        rescale_inputs: bool = True,        
        dropout_rate: Optional[float] = None,
    ):
    """
    build a convmixer network as described in the original paper "patches are all you need"
    i've added some optional dropout layers that could potentially help a bit in case you are training from scratch with a limited amount of data

    Args:
        input_shape (Tuple[int, int, int]): image input shape (height, width, channels)
        patches_embedding_dimension (int): hidden dimension, more specifically the size of patches embedding (described as "h" in the paper). This is also used as number of filter in the pointwise convolution of convmixer block.
        depth (int): depth of convmixer network, define the number of repetitions of convmixer block (described as "d" in the paper)
        patch_size (int): patch size used when breaking down input image in patches (described as "p" in the paper)
        kernel_size (int | Tuple[int, int]): kernel size of the depthwise convolution (described as "k" in the paper)
        num_classes (int): number of classes to classify images into
        rescale_inputs (bool, optional): optionally rescale the inputs between 0 and 1. Default to True.
        dropout_rate (bool, optional): dropout rate for optional dropouts layer, this is not part of the original paper implementation. Default to None. 

    Returns:
        keras.Model: convmixer keras model
    """
    
    # define inputs
    inputs = keras.Input(shape=input_shape, name='input')

    if rescale_inputs:
        inputs = keras.layers.Rescaling(scale=1./255)(inputs)

    # patches embedding through convolution
    patches = keras.layers.Conv2D(filters=patches_embedding_dimension, kernel_size=patch_size, strides=patch_size, name='patches')(inputs)
    patches = keras.layers.Activation('gelu', name='patches-gelu')(patches)
    layer = keras.layers.BatchNormalization(name='patches-batchnorm')(patches)

    # sequence of convmixer blocks
    for i in range(depth):
        if dropout_rate is not None:
            layer = keras.layers.Dropout(rate=dropout_rate, name=f'convmixer-block{i}-dropout')(layer)            
        layer = blocks.convmixer_block(
            layer=layer,
            name_prefix=f'convmixer-block{i}',
            filters=patches_embedding_dimension,
            kernel_size=kernel_size
        )

    # apply global average pooling to get a feature vector
    layer = keras.layers.GlobalAveragePooling2D(name='global-avg-pool')(layer)

    # classification head
    if dropout_rate is not None:
        layer = keras.layers.Dropout(rate=dropout_rate, name='dropout')(layer)
    layer = keras.layers.Dense(units=num_classes, name='dense')(layer)
    outputs = keras.layers.Softmax(name='softmax')(layer)

    # define model
    model = keras.Model(inputs, outputs)
    
    return model
