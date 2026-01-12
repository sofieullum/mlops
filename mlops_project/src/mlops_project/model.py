import torch
from torch import nn


class Model(nn.Module):
    """JA convolutional neural netork with a forward function"""
    def __init__(self, conv1_chnls=32, conv2_chnls=64, conv3_chnls=128, dr=0.5, num_classes=10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, conv1_chnls, 3, 1)
        self.conv2 = nn.Conv2d(conv1_chnls, conv2_chnls, 3, 1)
        self.conv3 = nn.Conv2d(conv2_chnls, conv3_chnls, 3, 1)

        self.dropout = nn.Dropout(p=dr)

        self.fc1 = nn.Linear(conv3_chnls, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2, 2)

        x = torch.flatten(x, 1)

        x = self.dropout(x)

        x = self.fc1(x)

        return x


if __name__ == "__main__":
    model = Model()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
