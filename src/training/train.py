# Code pour lancer un entraînement complet


from pathlib import Path
import json
import os
import sys
sys.path.append(os.path.abspath(".."))
import tensorflow as tf
import pandas as pd


from src.data.loaders import load_cifar10, split_train_val
from src.models.resnet20 import build_resnet20


def main():
    
    # Configuration des hyperparamètres baseline

    seed = 42
    epochs = 5
    batch_size = 32
    learning_rate = 0.1
    momentum = 0.9

    tf.random.set_seed(seed)

    # Configuration de la run

    run_config = {
        "model": "ResNet20",
        "dataset": "CIFAR10",
        "initializer": "He",
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "optimizer": "SGD",
        "learning_rate": learning_rate,
        "momentum": momentum,
    }
    output_dir = Path("runs/baseline_resnet20_cifar10_he")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=4)

    # Chargement des données

    (x_train, y_train), (x_test, y_test) = load_cifar10()
    (x_train, y_train), (x_val, y_val) = split_train_val(x_train, y_train, val_ratio=0.1, seed=42)

    print(x_train.shape, y_train.shape)
    print(x_val.shape, y_val.shape)
    print(x_test.shape, y_test.shape)

    # Construction du modèle 

    model = build_resnet20(input_shape=(32, 32, 3), num_classes=10)

    # Compilation du modèle 

    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate,
        momentum=momentum
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Entraînement du modèle 

    print("===================================")
    print("Run configuration")
    print("===================================")
    for key, value in run_config.items():
        print(f"{key}: {value}")
    print("===================================")

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    
    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", range(1, len(history_df) + 1))
    history_df.to_csv(output_dir / "history.csv", index=False)

    # Évaluation du modèle

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)

    summary = {
        "final_train_loss": float(history.history["loss"][-1]),
        "final_train_accuracy": float(history.history["accuracy"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "final_val_accuracy": float(history.history["val_accuracy"][-1]),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Outputs saved in: {output_dir}")

    # -----------------------------
    # 7. Sauvegarde minimale
    # -----------------------------
    output_dir = Path("runs/baseline_resnet20_cifar10_he")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history.history, f, indent=4)

    model.save(output_dir / "model.keras")

    print("Run terminé avec succès.")
    print(f"Historique sauvegardé dans : {output_dir / 'history.json'}")


if __name__ == "__main__":
    main()