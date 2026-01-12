import torch
import typer
from data_solution import corrupt_mnist
from model import MyAwesomeModel
import matplotlib.pyplot as plt

app = typer.Typer()


@app.command()
def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 10) -> None:
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    train_set, _ = corrupt_mnist()
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

            accuracy = (y_pred.argmax(dim = 1) == labels).float().mean().item()

            train_accuracy.append(accuracy)

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")
    
    print("Done training")
    torch.save(model.state_dict(), "model.pth")

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(train_losses)
    ax[0].set_title("Training loss")
    ax[1].plot(train_accuracy)
    ax[1].set_title("Training accuracy")
    fig.savefig("training_stats.png")

            



@app.command()
def evaluate(model_checkpoint: str) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depends on it")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = MyAwesomeModel()
    model.load_state_dict(torch.load(model_checkpoint))

    _, test_set = corrupt_mnist()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size = 32)

    model.eval()

    correct, total = 0, 0

    
    for images, labels in test_dataloader:
        y_pred = model(images)
        correct += (y_pred.argmax(dim = 1) == labels).float().sum().item()

        total += labels.size(0)

    accuracy = correct/total   

    print(f"The test accuracy is {accuracy}")      


if __name__ == "__main__":
    app()
