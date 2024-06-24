import keras
from keras import models, layers


def get_base_encoder(input_shape:tuple, name:str="VGG16", **kwargs):
    """Get a pretrained Keras model from `keras.applications`.

    Parameters
    ----------
    input_shape
        shape of the input
    name, optional
        name of pretrained model, by default "VGG16"
    """
    # shape adaptation layer to handle the case the number of channels <> 3
    if input_shape[-1] != 3:
        # If the number of channels in the orignal data isn't 3, use a first layer to adapt to the base model
        input_model = models.Sequential([
            layers.Input(shape=input_shape, name='input'),
            layers.Conv2D(3, kernel_size=(1,1), activation=None, padding='same')
        ])
        # input_shape = input_model(layers.Input(input_shape)).shape
        input_shape1 = (*input_shape[:-1], 3)
    else:
        input_model = models.Sequential([
            layers.Input(shape=input_shape, name='input'),
        ])
        input_shape1 = input_shape

    # config for the network
    base_model = getattr(keras.applications, name)(input_shape=input_shape1, **kwargs)

    x = layers.Input(input_shape)
    y = base_model(input_model(x))

    return models.Model(x, y)