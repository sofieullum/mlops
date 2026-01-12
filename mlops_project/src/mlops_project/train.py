import hydra
import matplotlib.pyplot as plt
import torch
import typer

from src.mlops_project.data import corrupt_mnist
from src.mlops_project.model import Model
from omegaconf import DictConfig
from pathlib import Path


@hydra.main(version_base = None, config_path="../../configs", config_name="config")
def train(cfg: DictConfig) -> None:
    """Trains the model and saves it along with the loss and accuracy. Takes learning rate, batch size, and
    epochs as an input"""
    hparams = cfg
    lr = cfg.learning_rate #hparams["learning_rate"]
    batch_size = hparams["batch_size"]
    epochs = hparams["epochs"]
    train_set, _ = corrupt_mnist()
    model = Model(conv1_chnls = cfg.conv1_channels,
        conv2_chnls = cfg.conv2_channels,
        conv3_chnls = cfg.conv3_channels,
        dr = cfg.dropout_rate,
        num_classes = cfg.num_classes
        )
    # add rest of your training code here
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, train_accuracy = [], []

    for epoch in range(epochs):
        model.train()
        for i, (images, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()

            y_pred = model(images)
            loss = criterion(y_pred, labels)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            accuracy = (y_pred.argmax(dim=1) == labels).float().mean().item()

            train_accuracy.append(accuracy)

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

    print("Done training")
    Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "models/model.pth")

    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(train_losses)
    ax[0].set_title("Training loss")
    ax[1].plot(train_accuracy)
    ax[1].set_title("Training accuracy")
    fig.savefig("reports/figures/training_stats.png")


if __name__ == "__main__":
    train()
