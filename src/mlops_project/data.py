
import torch
import typer


def normalize(images: torch.Tensor) -> torch.Tensor:
    """Returns the normalized images"""
    return (images - images.mean()) / images.std()


def preprocess(raw_dir: str, proccesed_dir: str):
    """Preprocesses the images, so they are normalized"""

    train_images = []
    train_target = []

    for i in range(6):
        train_images.append(torch.load(f"{raw_dir}/train_images_{i}.pt", weights_only=True))
        train_target.append(torch.load(f"{raw_dir}/train_target_{i}.pt", weights_only=True))

    train_images = torch.cat(train_images)
    train_target = torch.cat(train_target)

    test_images = torch.load(f"{raw_dir}/test_images.pt", weights_only=True)
    test_target = torch.load(f"{raw_dir}/test_target.pt", weights_only=True)

    train_images = train_images.unsqueeze(1).float()
    test_images = test_images.unsqueeze(1).float()

    train_images = normalize(train_images)
    test_images = normalize(test_images)

    train_target = train_target.long()
    test_target = test_target.long()

    torch.save(train_images, f"{proccesed_dir}/train_images.pt")
    torch.save(train_target, f"{proccesed_dir}/train_target.pt")
    torch.save(test_images, f"{proccesed_dir}/test_images.pt")
    torch.save(test_target, f"{proccesed_dir}/test_target.pt")


def corrupt_mnist():
    """Loads the preprocessed images"""
    path = "data/processed"
    train_images = torch.load(f"{path}/train_images.pt")
    train_target = torch.load(f"{path}/train_target.pt")
    test_images = torch.load(f"{path}/test_images.pt")
    test_target = torch.load(f"{path}/test_target.pt")

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)

    return train_set, test_set


if __name__ == "__main__":
    typer.run(preprocess)
