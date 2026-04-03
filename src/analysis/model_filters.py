import numpy as np
import tensorflow as tf


# Retourne la liste des couches Conv2D 3x3 d'un modèle Keras
# Ignore notamment les convolutions 1x1 (shortcuts)


def get_conv3x3_layers(model) -> list:

    conv3x3_layers = []

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D) and layer.kernel_size == (3, 3):
            conv3x3_layers.append(layer)

    return conv3x3_layers


# Retourne les filtres d'une couche Conv2D 3x3 donnée par son index dans la liste filtrée
# Cela permettra d'analyser a posteriori les filtres d'une couche spécifique (ex: première couche, couche centrale, etc.)
def get_layer_filters(model, layer_idx: int) -> np.ndarray:

    conv3x3_layers = get_conv3x3_layers(model)

    if layer_idx < 0 or layer_idx >= len(conv3x3_layers):
        raise IndexError(
            f"layer_idx={layer_idx} is out of range for {len(conv3x3_layers)} Conv2D 3x3 layers."
        )

    layer = conv3x3_layers[layer_idx]
    weights = layer.get_weights()

    if len(weights) == 0:
        raise ValueError(f"Layer '{layer.name}' has no weights.")

    filters = np.asarray(weights[0], dtype=np.float64)
    
    if filters.ndim != 4:
        raise ValueError(
            f"Expected filters to have 4 dimensions, got shape {filters.shape}"
        )

    if filters.shape[0] != 3 or filters.shape[1] != 3:
        raise ValueError(
            f"Expected filter spatial shape (3, 3, in_channels, out_channels), got {filters.shape}"
        )

    return filters



# Itérer sur tous les noyaux d'un modèle
# Pour rappel, le plus petit niveau d'analyse est le noyau 3x3 d'une convolution, cette fonction permet de travailler à ce niveau
def iter_conv_kernels(model):

    conv3x3_layers = get_conv3x3_layers(model)

    for layer_idx, layer in enumerate(conv3x3_layers):
        filters = get_layer_filters(model, layer_idx)
        _, _, in_channels, out_channels = filters.shape

        for in_channel in range(in_channels):
            for out_channel in range(out_channels):
                kernel = filters[:, :, in_channel, out_channel]

                yield {
                    "layer_idx": layer_idx,
                    "layer_name": layer.name,
                    "in_channel": in_channel,
                    "out_channel": out_channel,
                    "kernel": kernel,
                }