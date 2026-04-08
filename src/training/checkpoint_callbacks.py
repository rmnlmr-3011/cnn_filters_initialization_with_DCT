# Code pour sauvegarder le modèle à certains epochs choisis, au début et à la fin de l'entraînement.



from pathlib import Path
from typing import Iterable, Union, Optional

import tensorflow as tf


class EpochCheckpointCallback(tf.keras.callbacks.Callback):

    def __init__(
        self,
        checkpoint_dir: Union[Path, str],
        save_epochs: Optional[Iterable[int]] = None,
        save_initial: bool = True,
        save_final: bool = True,
        verbose: bool = True,
    ):
        super().__init__()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.save_epochs = set(save_epochs or [])
        self.save_initial = save_initial
        self.save_final = save_final
        self.verbose = verbose

    def _save_model(self, filename: str):
        path = self.checkpoint_dir / filename
        self.model.save(path)

        if self.verbose:
            print(f"\n[Checkpoint] Saved: {path}")

    def on_train_begin(self, logs=None):
        if self.save_initial:
            self._save_model("epoch_000.keras")

    def on_epoch_end(self, epoch, logs=None):
        epoch_completed = epoch + 1  # Keras donne epoch=0 pour le premier epoch terminé

        if epoch_completed in self.save_epochs:
            self._save_model(f"epoch_{epoch_completed:03d}.keras")

    def on_train_end(self, logs=None):
        if self.save_final:
            self._save_model("epoch_final.keras")