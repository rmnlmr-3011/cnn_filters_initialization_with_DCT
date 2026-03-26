# Code pour évaluer un modèle entrainé

import tensorflow as tf


def evaluate_model(model: tf.keras.Model, x_test, y_test):
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    return {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
    }