# # Code à éxécuter pour lancer un entraînement complet (sans l'analyse derrière)




from src.training.train_core import run_training


def main():
    run_training("configs/base.yaml")


if __name__ == "__main__":
    main()