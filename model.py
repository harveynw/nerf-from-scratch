import torch
from torch import nn

T = torch.tensor


class PositionalEncoding(nn.Module):
    def __init__(self, L: int):
        super(PositionalEncoding, self).__init__()
        self.L = L

    def forward(self, vectors: T) -> T:
        # Positional Encoding: Eq (4) γ(p), from the original paper
        elements = [vectors] 
        for i in range(self.L):
            elements += [torch.sin(2.0**i * torch.pi * vectors), torch.cos(2.0**i * torch.pi * vectors)]
        output = torch.concat(elements, dim=1)
        return output

    def output_dim(self) -> int:
        return 3 + self.L * 2 * 3


# Debug NeRF model with no trainable parameters, outputs shapes that are good for testing the render.py equation
class DebugNeRF(nn.Module):
    def __init__(self, mode: str = 'balls'):
        super(DebugNeRF, self).__init__()
        self.mode = mode

    def forward(self, position: T, direction: T) -> (T, T):
        if self.mode == 'balls':
            batch_size = position.shape[0]
            x, y, z = position[:, 0], position[:, 1], position[:, 2]

            volume_density = torch.zeros(batch_size)
            radiance = torch.zeros(batch_size, 3)

            inside_ball_1 = torch.square(x-0.4) + torch.square(y+0.4) + torch.square(z) < 0.1
            inside_ball_2 = torch.square(x+0.5) + torch.square(y+0.2) + torch.square(z) < 0.8
            inside_balls = inside_ball_1 | inside_ball_2

            volume_density[inside_balls] = 1.0
            radiance[inside_ball_1, :] = torch.tensor([1.0, 0.0, 0.0])
            radiance[inside_ball_2, :] = torch.tensor([0.0, 1.0, 0.0])

            return volume_density, radiance
        elif self.mode == 'center_balls':
            batch_size = position.shape[0]
            x, y, z = position[:, 0], position[:, 1], position[:, 2]

            volume_density = torch.zeros(batch_size)
            radiance = torch.zeros(batch_size, 3)

            size = 0.1
            inside_ball_1 = torch.square(x-1.0) + torch.square(y-1.0) + torch.square(z) < size
            inside_ball_2 = torch.square(x+1.0) + torch.square(y-1.0) + torch.square(z) < size
            inside_ball_3 = torch.square(x-1.0) + torch.square(y+1.0) + torch.square(z) < size
            inside_ball_4 = torch.square(x+1.0) + torch.square(y+1.0) + torch.square(z) < size

            inside_balls = inside_ball_1 | inside_ball_2 | inside_ball_3 | inside_ball_4

            volume_density[inside_balls] = 1.0
            radiance[inside_ball_1] = torch.tensor([1.0, 0.0, 0.0])
            radiance[inside_ball_2] = torch.tensor([0.0, 1.0, 0.0])
            radiance[inside_ball_3] = torch.tensor([0.0, 0.0, 1.0])
            radiance[inside_ball_4] = torch.tensor([1.0, 1.0, 1.0])

            return volume_density, radiance
              
        return None, None


# Implementing Figure 7
class NeRF(nn.Module):  
    def __init__(self):
        super(NeRF, self).__init__()

        self.enc_position = PositionalEncoding(L=10)
        self.enc_direction = PositionalEncoding(L=4)

        position_input_dim = self.enc_position.output_dim()
        direction_input_dim = self.enc_direction.output_dim()

        self.block_1 = nn.Sequential(
            nn.Linear(position_input_dim, 256),
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
            nn.Linear(256 + position_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Linear(256, 1 + 256),
        )
        self.block_3 = nn.Sequential(
            nn.Linear(256 + direction_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Sigmoid()
        )

    def forward(self, position: T, direction: T) -> (T, T):
        p, d = self.enc_position(position), self.enc_direction(direction)

        x_1 = self.block_1(p)
        x_2 = self.block_2(torch.cat([x_1, p], dim=1))

        volume_density = torch.relu(x_2[:, 0])
        radiance = self.block_3(torch.cat([x_2[:, 1:(256+1)], d], dim=1))
  
        return volume_density, radiance

# Much smaller model, faster to train
class TinyNeRF(nn.Module):
    def __init__(self): 
        super(TinyNeRF, self).__init__()
        
        self.enc_position = PositionalEncoding(L=6)
        input_dim = self.enc_position.output_dim()
 
        self.block_1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        ) 
        self.block_2 = nn.Sequential(
            nn.Linear(256 + input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(256, 4)

    def forward(self, position: T, direction: T) -> (T, T):
        x = self.enc_position(position)
        y = torch.concat([self.block_1(x), x], dim=1)
        z = self.output_layer(self.block_2(y))
        return torch.relu(z[:, 3]), torch.sigmoid(z[:, :3])
