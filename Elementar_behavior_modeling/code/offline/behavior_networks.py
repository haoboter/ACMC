"""Behavior modeling networks; load_state_dict must match class and architecture."""
import torch
import torch.nn as nn


class NN_motion_prediction(nn.Module):
    """Scaled 4D field inputs -> scalar; final tanh, output in (-1, 1). Shared by milli / nano velocity."""

    def __init__(self, input_dim: int = 4) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128, track_running_stats=True)
        self.fc2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256, track_running_stats=True)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128, track_running_stats=True)
        self.dropout = nn.Dropout(0.5)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        return torch.tanh(self.fc4(x))


class NN_release_rate_prediction(nn.Module):
    """Field inputs -> Final_Gray: sigmoid then *255, range [0, 255]."""

    def __init__(self, input_dim: int = 4) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128, track_running_stats=True)
        self.fc2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256, track_running_stats=True)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128, track_running_stats=True)
        self.dropout = nn.Dropout(0.5)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        return torch.sigmoid(self.fc4(x)) * 255.0
