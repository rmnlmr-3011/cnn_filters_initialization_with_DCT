import tensorflow as tf


def make_multistep_scheduler(
    initial_lr: float,
    milestones,
    gamma: float = 0.1,
):
    """
    Scheduler MultiStep à milestones absolus.

    Exemple:
        initial_lr = 0.1
        milestones = [3, 5]
        gamma = 0.1

    Donne :
        epochs < 3   -> 0.1
        3 <= epoch < 5 -> 0.01
        epoch >= 5   -> 0.001

    Remarque:
    - `epoch` fourni par Keras est 0-indexé
    - on convertit en epoch humain avec `epoch + 1`
    """
    milestones = sorted(list(milestones))

    def lr_schedule(epoch, lr):
        epoch_num = epoch + 1  # epochs humains: 1, 2, 3, ...

        factor = 1.0
        for milestone in milestones:
            if epoch_num >= milestone:
                factor *= gamma

        return initial_lr * factor

    return tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)


class LearningRateLogger(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        print(f"\n[LR] Epoch {epoch+1}: {lr:.6f}")

# # Code pour le scheduler


# import tensorflow as tf


# def make_multistep_scheduler(initial_lr: float, total_epochs: int):
#     """
#     Scheduler MultiStep simple :
#     - 0% à 50%   : lr = initial_lr
#     - 50% à 75%  : lr = initial_lr * 0.1
#     - 75% à 100% : lr = initial_lr * 0.01
#     """

#     def lr_schedule(epoch, lr):
#         if epoch < 0.5 * total_epochs:
#             return initial_lr
#         elif epoch < 0.75 * total_epochs:
#             return initial_lr * 0.1
#         else:
#             return initial_lr * 0.01

#     return tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)


# class LearningRateLogger(tf.keras.callbacks.Callback):
#     def on_epoch_begin(self, epoch, logs=None):
#         lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
#         print(f"\n[LR] Epoch {epoch+1}: {lr:.6f}")

