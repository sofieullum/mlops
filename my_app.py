import hydra
from omegaconf import DictConfig
from src.mlops_project.model import Model
from src.mlops_project.train import train
from src.mlops_project.evaluate import evaluate

@hydra.main(config_path="config", config_name="config.yaml")
def main(cfg: DictConfig):
