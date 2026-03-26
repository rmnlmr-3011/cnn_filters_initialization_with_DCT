# Code pour le scheduler


import tensorflow as tf


def make_multistep_scheduler(initial_lr: float, total_epochs: int):
    """
    Scheduler MultiStep simple :
    - 0% à 50%   : lr = initial_lr
    - 50% à 75%  : lr = initial_lr * 0.1
    - 75% à 100% : lr = initial_lr * 0.01
    """

    def lr_schedule(epoch, lr):
        if epoch < 0.5 * total_epochs:
            return initial_lr
        elif epoch < 0.75 * total_epochs:
            return initial_lr * 0.1
        else:
            return initial_lr * 0.01

    return tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)


class LearningRateLogger(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        print(f"\n[LR] Epoch {epoch+1}: {lr:.6f}")