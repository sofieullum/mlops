import hydra
from omegaconf import DictConfig


@hydra.main(config_path="config", config_name="config.yaml")
def main(cfg: DictConfig):
    return
