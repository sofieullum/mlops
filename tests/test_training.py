import os

from hydra import compose, initialize_config_dir


def test_train():
    # Get absolute path to configs
    config_path = os.path.abspath("configs")

    with initialize_config_dir(version_base=None, config_dir=config_path):
        cfg = compose(config_name="config", overrides=["epochs=2"])

        # Import here to avoid Hydra initialization issues
        from src.mlops_project.train import train
        train(cfg)
