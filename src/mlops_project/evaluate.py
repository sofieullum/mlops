import torch
import typer

from mlops_project.data import corrupt_mnist
from mlops_project.model import Model


def evaluate(model_checkpoint: str) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depends on it")
    print(model_checkpoint)

    model = Model()
    model.load_state_dict(torch.load(model_checkpoint, weights_only=True))

    _, test_set = corrupt_mnist()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)

    model.eval()

    correct, total = 0, 0

    for images, labels in test_dataloader:
        y_pred = model(images)
        correct += (y_pred.argmax(dim=1) == labels).float().sum().item()

        total += labels.size(0)

    accuracy = correct / total

    print(f"The test accuracy is {accuracy}")


if __name__ == "__main__":
    typer.run(evaluate)
