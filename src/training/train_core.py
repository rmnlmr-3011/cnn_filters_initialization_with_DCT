# Code contenant la logique d'entraînement du modèle, avec gestion de la configuration, des callbacks, et de la sauvegarde des résultats.


from pathlib import Path
from typing import Union

import tensorflow as tf

from src.data.loaders import load_cifar10, split_train_val
from src.models.resnet20 import build_resnet20
from src.training.callbacks import make_multistep_scheduler, LearningRateLogger
from src.training.checkpoint_callbacks import EpochCheckpointCallback
from src.training.evaluate import evaluate_model
from src.utils.config import load_config
from src.utils.io import save_json, save_history_csv, make_run_dir
from src.utils.logging import get_logger
from src.utils.seed import set_seed
from src.models.initializers import get_initializer



def _build_initializer_from_spec(spec: dict, global_seed: int = None):
    """
    Construit un initializer Keras à partir d'un bloc YAML du type :
    {
        "name": "dctlow_noise",
        "scale": 1.0,
        "noise_std": 0.05,
        "seed": 42,
    }
    """
    if "name" not in spec:
        raise ValueError(f"Initializer spec missing 'name': {spec}")

    init_name = spec["name"]
    init_seed = spec.get("seed", global_seed)

    kwargs = {
        k: v
        for k, v in spec.items()
        if k not in ("name", "seed", "mode")
    }

    return get_initializer(init_name, seed=init_seed, **kwargs)


def _resolve_model_initialization(config: dict):
    """
    Retourne :
    - conv_initializer
    - layer_init_map

    Cas supportés :
    1) initializer global
    2) initializer per_layer
    """
    seed = config.get("seed", None)
    initializer_cfg = config.get("initializer", {})

    if not initializer_cfg:
        # fallback par défaut
        return get_initializer("he", seed=seed), None

    mode = initializer_cfg.get("mode", "global")

    if mode == "global":
        conv_initializer = _build_initializer_from_spec(initializer_cfg, global_seed=seed)
        return conv_initializer, None

    if mode == "per_layer":
        if "default" not in initializer_cfg:
            raise ValueError("Per-layer initializer config requires a 'default' block.")

        default_initializer = _build_initializer_from_spec(
            initializer_cfg["default"],
            global_seed=seed
        )

        raw_layer_map = initializer_cfg.get("layer_map", {})
        layer_init_map = {
            layer_name: _build_initializer_from_spec(layer_spec, global_seed=seed)
            for layer_name, layer_spec in raw_layer_map.items()
        }

        return default_initializer, layer_init_map

    raise ValueError(f"Unsupported initializer mode: {mode}")

def _evaluate_split(model: tf.keras.Model, x, y, split_name: str, verbose: int = 0) -> dict:
    """
    Évalue le modèle sur un split donné et retourne un dict homogène.
    """
    loss, accuracy = model.evaluate(x, y, verbose=verbose)
    return {
        f"{split_name}_loss": float(loss),
        f"{split_name}_accuracy": float(accuracy),
    }

def run_training(config_path: Union[str, Path] = "configs/base.yaml") -> Path:
    """
    Exécute un entraînement complet et retourne le dossier du run créé.

    Cette fonction :
    - charge la config
    - crée le dossier de run
    - entraîne le modèle
    - sauvegarde les sorties classiques :
        - config.json
        - history.csv
        - summary.json
        - model.keras
    - sauvegarde aussi les checkpoints intermédiaires si activés

    Parameters
    ----------
    config_path:
        Chemin vers le fichier YAML de configuration.

    Returns
    -------
    Path
        Le chemin du dossier de run créé.
    """
    config_path = Path(config_path)
    config = load_config(config_path)

    seed = config["seed"]

    training_cfg = config["training"]




    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    runtime_cfg = config["runtime"]
    checkpoint_cfg = config.get("checkpoints", {})
    early_stopping_cfg = training_cfg.get("early_stopping", {})
    scheduler_cfg = training_cfg.get("scheduler", {})


    epochs = training_cfg["epochs"]
    batch_size = training_cfg["batch_size"]
    learning_rate = training_cfg["learning_rate"]
    momentum = training_cfg["momentum"]

    early_stopping_cfg = training_cfg.get("early_stopping", {})
    early_stopping_enabled = early_stopping_cfg.get("enabled", False)
    early_stopping_monitor = early_stopping_cfg.get("monitor", "val_loss")
    early_stopping_patience = early_stopping_cfg.get("patience", 10)
    early_stopping_min_delta = early_stopping_cfg.get("min_delta", 0.0)
    early_stopping_restore_best_weights = early_stopping_cfg.get("restore_best_weights", True)

    validation_split = dataset_cfg["validation_split"]
    normalize = dataset_cfg["normalize"]

    input_shape = tuple(model_cfg["input_shape"])
    num_classes = model_cfg["num_classes"]

    run_root = runtime_cfg["run_root"]
    verbose = runtime_cfg["verbose"]
    deterministic = runtime_cfg["deterministic"]

    scheduler_milestones = scheduler_cfg.get("milestones", [3, 5])
    scheduler_gamma = scheduler_cfg.get("gamma", 0.1)

    checkpoint_enabled = checkpoint_cfg.get("enabled", False)
    checkpoint_save_initial = checkpoint_cfg.get("save_initial", True)
    checkpoint_save_final = checkpoint_cfg.get("save_final", True)
    checkpoint_save_epochs = checkpoint_cfg.get("save_epochs", [])

    run_config = config

    set_seed(seed, deterministic=deterministic)

    output_dir = make_run_dir(
        base_dir=run_root,
        prefix=config["experiment_name"]
    )

    logger = get_logger("train")
    save_json(run_config, output_dir / "config.json")

    # =========================
    # Chargement des données
    # =========================
    (x_train, y_train), (x_test, y_test) = load_cifar10(normalize=normalize)
    (x_train, y_train), (x_val, y_val) = split_train_val(
        x_train,
        y_train,
        val_ratio=validation_split,
        seed=seed
    )

    logger.info(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
    logger.info(f"x_val: {x_val.shape}, y_val: {y_val.shape}")
    logger.info(f"x_test: {x_test.shape}, y_test: {y_test.shape}")

    # =========================
    # Construction du modèle
    # =========================

    conv_initializer, layer_init_map = _resolve_model_initialization(config)

    model = build_resnet20(
        input_shape=input_shape,
        num_classes=num_classes,
        conv_initializer=conv_initializer,
        layer_init_map=layer_init_map,
    )
    

    # Ici, plus tard si nécessaire :
    # model = init_filters(model, config)

    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate,
        momentum=momentum
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    # =========================
    # Callbacks
    # =========================
    lr_scheduler = make_multistep_scheduler(
        initial_lr=learning_rate,
        milestones=scheduler_milestones,
        gamma=scheduler_gamma,
    )

    callbacks = [
        lr_scheduler,
        LearningRateLogger(),
    ]

    if checkpoint_enabled:
        callbacks.append(
            EpochCheckpointCallback(
                checkpoint_dir=output_dir / "checkpoints",
                save_epochs=checkpoint_save_epochs,
                save_initial=checkpoint_save_initial,
                save_final=checkpoint_save_final,
                verbose=bool(verbose),
            )
        )
    
    if early_stopping_enabled:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=early_stopping_monitor,
                patience=early_stopping_patience,
                min_delta=early_stopping_min_delta,
                restore_best_weights=early_stopping_restore_best_weights,
                mode="min",
                verbose=1,
            )
        )

    # =========================
    # Logs config
    # =========================
    logger.info("===================================")
    logger.info("Run configuration")
    logger.info("===================================")
    for key, value in run_config.items():
        logger.info(f"{key}: {value}")
    logger.info("===================================")

    # =========================
    # Entraînement
    # =========================


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

    # =========================
    # Évaluation finale cohérente
    # =========================
    train_metrics = _evaluate_split(model, x_train, y_train, "final_train", verbose=0)
    val_metrics = _evaluate_split(model, x_val, y_val, "final_val", verbose=0)
    test_metrics = evaluate_model(model, x_test, y_test)

    summary = {
        **train_metrics,
        **val_metrics,
        **test_metrics,
    }

    save_json(summary, output_dir / "summary.json")
    model.save(output_dir / "model.keras")

    logger.info(f"Final train loss: {summary['final_train_loss']:.4f}")
    logger.info(f"Final train accuracy: {summary['final_train_accuracy']:.4f}")
    logger.info(f"Final val loss: {summary['final_val_loss']:.4f}")
    logger.info(f"Final val accuracy: {summary['final_val_accuracy']:.4f}")
    logger.info(f"Test loss: {summary['test_loss']:.4f}")
    logger.info(f"Test accuracy: {summary['test_accuracy']:.4f}")
    logger.info(f"Outputs saved in: {output_dir}")
    logger.info("Run terminé avec succès.")

    return output_dir

    