import torch

PATH = "data/corruptmnist_v1"

def corrupt_mnist():
    """Return train and test dataloaders for corrupt MNIST."""
    # exchange with the corrupted mnist dataset
    train_images = []
    train_target = []

    for i in range(6):
        train_images.append(torch.load(f"{PATH}/train_images_{i}.pt", weights_only=True))
        train_target.append(torch.load(f"{PATH}/train_target_{i}.pt", weights_only=True))

    train_images = torch.cat(train_images)
    train_target = torch.cat(train_target)

    test_images = torch.load(f"{PATH}/test_images.pt", weights_only=True)
    test_target = torch.load(f"{PATH}/test_target.pt", weights_only=True)

    train_images = train_images.unsqueeze(1).float()
    test_images = test_images.unsqueeze(1).float()

    train_target = train_target.long()
    test_target = test_target.long()

    train_set = torch.utils.data.TensorDataset(train_images,train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)
    return train_set, test_set


if __name__ == "__main__":
    train_set, test_set = corrupt_mnist()
    print(f"Size of training set: {len(train_set)}")
    print(f"Size of test set: {len(test_set)}")
    print(f"Shape of a training point {(train_set[0][0].shape, train_set[0][1].shape)}")
    print(f"Shape of a test point {(test_set[0][0].shape, test_set[0][1].shape)}")
