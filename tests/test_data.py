import os.path

import pytest
import torch

from src.mlops_project.data import corrupt_mnist


@pytest.mark.skipif(not os.path.exists("data/processed/train_images.pt"), reason="Training images not found")
@pytest.mark.skipif(not os.path.exists("data/processed/train_target.pt"), reason="Training targets not found")
@pytest.mark.skipif(not os.path.exists("data/processed/test_images.pt"), reason="Test images not found")
@pytest.mark.skipif(not os.path.exists("data/processed/test_target.pt"), reason="Test targets not found")
def test_my_dataset():
    """Test the MyDataset class."""
    train, test = corrupt_mnist()
    assert len(train) == 30000, f"Expected train set of length 30000, but got length {len(train)}"
    assert len(test) == 5000, f"Expected test set of length 50000, but got length {len(test)}"
    for dataset in [train, test]:
        for x,y in dataset:
            assert x.shape == torch.Size([1,28,28]), f"Expected x.shape==torch.Size([1,28,28]), but got {x.shape}"
            assert y in range (10), f"Expected y to be in range 10, but got {y}"
    train_targets = torch.unique(train.tensors[1])
    assert (train_targets == torch.arange(0,10)).all(), (
    f"Expected training targets to be in range 1-10, got {train_targets}"
)
    test_targets = torch.unique(test.tensors[1])
    assert (test_targets == torch.arange(0,10)).all(), (
    f"Expected test targets to be in range 1-10, got {test_targets}"
)


if __name__ == "__main__":
    test_my_dataset()
