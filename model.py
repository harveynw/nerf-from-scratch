import torch
from torch import nn

T = torch.tensor


class NeRF(nn.Module):  # Implementing Figure 7
    def __init__(self):
        super(NeRF, self).__init__()

        self.block_1 = nn.Sequential(
            nn.Linear(60, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.block_2 = nn.Sequential(
            nn.Linear(60 + 256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Linear(256, 1 + 256),
        )
        self.block_3 = nn.Sequential(
            nn.Linear(256 + 24, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Sigmoid()
        )

    def forward(self, position_encoded: T, direction_encoded: T) -> (T, T):
        x_1 = self.block_1(position_encoded)
        x_2 = self.block_2(torch.cat([x_1, position_encoded], dim=1))

        volume_density = torch.relu(x_2[:, 0])
        radiance = self.block_3(torch.cat([x_2[:, 1:(256+1)], direction_encoded], dim=1))

        return volume_density, radiance
