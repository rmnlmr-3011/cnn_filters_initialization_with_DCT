import tensorflow as tf


def _resolve_initializer(layer_name, conv_initializer="he_normal", layer_init_map=None):
    """
    Retourne l'initializer à utiliser pour une couche donnée.
    Priorité :
    1) layer_init_map[layer_name] si présent
    2) conv_initializer sinon
    """
    if layer_init_map is not None and layer_name in layer_init_map:
        return layer_init_map[layer_name]
    return conv_initializer


def residual_block(
    x,
    filters,
    stride=1,
    conv_initializer="he_normal",
    layer_init_map=None,
    block_prefix="block",
):
    shortcut = x

    conv1_name = f"{block_prefix}_conv1"
    conv2_name = f"{block_prefix}_conv2"
    shortcut_name = f"{block_prefix}_shortcut"

    conv1_initializer = _resolve_initializer(
        conv1_name,
        conv_initializer=conv_initializer,
        layer_init_map=layer_init_map,
    )

    conv2_initializer = _resolve_initializer(
        conv2_name,
        conv_initializer=conv_initializer,
        layer_init_map=layer_init_map,
    )

    # Première convolution 3x3
    y = tf.keras.layers.Conv2D(
        filters,
        kernel_size=3,
        strides=stride,
        padding="same",
        use_bias=False,
        kernel_initializer=conv1_initializer,
        name=conv1_name,
    )(x)
    y = tf.keras.layers.BatchNormalization(name=f"{block_prefix}_bn1")(y)
    y = tf.keras.layers.ReLU(name=f"{block_prefix}_relu1")(y)

    # Deuxième convolution 3x3
    y = tf.keras.layers.Conv2D(
        filters,
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=False,
        kernel_initializer=conv2_initializer,
        name=conv2_name,
    )(y)
    y = tf.keras.layers.BatchNormalization(name=f"{block_prefix}_bn2")(y)

    # Shortcut 1x1 -> He
    if stride != 1 or x.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(
            filters,
            kernel_size=1,
            strides=stride,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
            name=shortcut_name,
        )(shortcut)
        shortcut = tf.keras.layers.BatchNormalization(name=f"{block_prefix}_shortcut_bn")(shortcut)

    out = tf.keras.layers.Add(name=f"{block_prefix}_add")([y, shortcut])
    out = tf.keras.layers.ReLU(name=f"{block_prefix}_relu_out")(out)

    return out


def build_resnet20(
    input_shape=(32, 32, 3),
    num_classes=10,
    conv_initializer="he_normal",
    layer_init_map=None,
):
    inputs = tf.keras.Input(shape=input_shape, name="input")

    stem_initializer = _resolve_initializer(
        "stem_conv",
        conv_initializer=conv_initializer,
        layer_init_map=layer_init_map,
    )

    # Stem
    x = tf.keras.layers.Conv2D(
        16,
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=False,
        kernel_initializer=stem_initializer,
        name="stem_conv",
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name="stem_bn")(x)
    x = tf.keras.layers.ReLU(name="stem_relu")(x)

    # Groupe 1
    x = residual_block(
        x, filters=16, stride=1,
        conv_initializer=conv_initializer,
        layer_init_map=layer_init_map,
        block_prefix="group1_block1",
    )
    x = residual_block(
        x, filters=16, stride=1,
        conv_initializer=conv_initializer,
        layer_init_map=layer_init_map,
        block_prefix="group1_block2",
    )
    x = residual_block(
        x, filters=16, stride=1,
        conv_initializer=conv_initializer,
        layer_init_map=layer_init_map,
        block_prefix="group1_block3",
    )

    # Groupe 2
    x = residual_block(
        x, filters=32, stride=2,
        conv_initializer=conv_initializer,
        layer_init_map=layer_init_map,
        block_prefix="group2_block1",
    )
    x = residual_block(
        x, filters=32, stride=1,
        conv_initializer=conv_initializer,
        layer_init_map=layer_init_map,
        block_prefix="group2_block2",
    )
    x = residual_block(
        x, filters=32, stride=1,
        conv_initializer=conv_initializer,
        layer_init_map=layer_init_map,
        block_prefix="group2_block3",
    )

    # Groupe 3
    x = residual_block(
        x, filters=64, stride=2,
        conv_initializer=conv_initializer,
        layer_init_map=layer_init_map,
        block_prefix="group3_block1",
    )
    x = residual_block(
        x, filters=64, stride=1,
        conv_initializer=conv_initializer,
        layer_init_map=layer_init_map,
        block_prefix="group3_block2",
    )
    x = residual_block(
        x, filters=64, stride=1,
        conv_initializer=conv_initializer,
        layer_init_map=layer_init_map,
        block_prefix="group3_block3",
    )

    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        kernel_initializer="he_normal",
        name="classifier",
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="ResNet20")
    return model