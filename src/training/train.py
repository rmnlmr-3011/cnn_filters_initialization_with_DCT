# Code pour lancer un entraînement complet


# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
# & "C:\Users\rmnlm\anaconda3\shell\condabin\conda-hook.ps1"
# conda activate sys809



from pathlib import Path
import tensorflow as tf

from src.data.loaders import load_cifar10, split_train_val
from src.models.resnet20 import build_resnet20
from src.training.callbacks import make_multistep_scheduler, LearningRateLogger
from src.utils.io import save_json, save_history_csv, make_run_dir
from src.utils.logging import get_logger
from src.utils.seed import set_seed
from src.utils.config import load_config
from src.training.evaluate import evaluate_model



def main():
    # Configuration des hyperparamètres baseline
    config = load_config("configs/base.yaml")

    seed = config["seed"]

    epochs = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    learning_rate = config["training"]["learning_rate"]
    momentum = config["training"]["momentum"]

    validation_split = config["dataset"]["validation_split"]
    normalize = config["dataset"]["normalize"]

    input_shape = tuple(config["model"]["input_shape"])
    num_classes = config["model"]["num_classes"]

    run_root = config["runtime"]["run_root"]
    verbose = config["runtime"]["verbose"]
    deterministic = config["runtime"]["deterministic"]

    run_config = config

    set_seed(seed, deterministic=deterministic)

    output_dir = make_run_dir(
        base_dir=run_root,
        prefix=config["experiment_name"]
    )

    logger = get_logger("train")
    save_json(run_config, output_dir / "config.json")

    # Chargement des données
    (x_train, y_train), (x_test, y_test) = load_cifar10(normalize=normalize)
    (x_train, y_train), (x_val, y_val) = split_train_val(x_train, y_train, val_ratio=validation_split, seed=seed)

    logger.info(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
    logger.info(f"x_val: {x_val.shape}, y_val: {y_val.shape}")
    logger.info(f"x_test: {x_test.shape}, y_test: {y_test.shape}")

    # Construction du modèle 
    model = build_resnet20(input_shape=input_shape, num_classes=num_classes)

    #model = initit_filters(model, config)

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

    # Callbacks
    lr_scheduler = make_multistep_scheduler(
        initial_lr=learning_rate,
        total_epochs=epochs
    )

    callbacks = [
        lr_scheduler,
        LearningRateLogger(),
    ]


    # Entraînement du modèle 
    logger.info("===================================")
    logger.info("Run configuration")
    logger.info("===================================")
    for key, value in run_config.items():
        logger.info(f"{key}: {value}")
    logger.info("===================================")

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose
    )
    
    save_history_csv(history, output_dir / "history.csv")


    # Évaluation du modèle
    test_metrics = evaluate_model(model, x_test, y_test)

    summary = {
        "final_train_loss": float(history.history["loss"][-1]),
        "final_train_accuracy": float(history.history["accuracy"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "final_val_accuracy": float(history.history["val_accuracy"][-1]),
        **test_metrics,
    }

    save_json(summary, output_dir / "summary.json")

    model.save(output_dir / "model.keras")

    logger.info(f"Test loss: {test_metrics['test_loss']:.4f}")
    logger.info(f"Test accuracy: {test_metrics['test_accuracy']:.4f}")
    logger.info(f"Outputs saved in: {output_dir}")
    logger.info("Run terminé avec succès.")


if __name__ == "__main__":
    main()