# Architecture du modèle Resnet20

import tensorflow as tf


def residual_block(x, filters, stride=1):
    shortcut = x

    # Première convolution
    y = tf.keras.layers.Conv2D(
        filters,
        kernel_size=3,
        strides=stride,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
    )(x)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.ReLU()(y)

    # Deuxième convolution
    y = tf.keras.layers.Conv2D(
        filters,
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
    )(y)
    y = tf.keras.layers.BatchNormalization()(y)

    # Projection du shortcut si dimensions différentes
    if stride != 1 or x.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(
            filters,
            kernel_size=1,
            strides=stride,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
        )(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    out = tf.keras.layers.Add()([y, shortcut])
    out = tf.keras.layers.ReLU()(out)

    return out





def build_resnet20(input_shape=(32, 32, 3), num_classes=10):
    inputs = tf.keras.Input(shape=input_shape)

    # Stem CIFAR
    x = tf.keras.layers.Conv2D(
        16,
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Groupe 1 : 3 blocs, 16 filtres
    for _ in range(3):
        x = residual_block(x, filters=16, stride=1)

    # Groupe 2 : 3 blocs, 32 filtres
    x = residual_block(x, filters=32, stride=2)
    for _ in range(2):
        x = residual_block(x, filters=32, stride=1)

    # Groupe 3 : 3 blocs, 64 filtres
    x = residual_block(x, filters=64, stride=2)
    for _ in range(2):
        x = residual_block(x, filters=64, stride=1)

    # Tête de classification
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        kernel_initializer="he_normal",
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="ResNet20")
    return model