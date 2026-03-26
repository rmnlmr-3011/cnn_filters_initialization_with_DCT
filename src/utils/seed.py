# On y fixera les seeds pour la reproductibilité

import os
import random
import numpy as np
import tensorflow as tf


def set_seed(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    if deterministic:
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass