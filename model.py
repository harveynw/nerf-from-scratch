import torch
from torch import nn

T = torch.tensor


class PositionalEncoding(nn.Module):
    def __init__(self, L: int):
        super(PositionalEncoding, self).__init__()
        self.L = L
        self.register_buffer(name='coef', tensor=torch.pow(2, torch.arange(start=0, end=self.L, step=1)) * torch.pi)

    def forward(self, vectors: T) -> T:
        dim = vectors.shape[1]
        rad = vectors.unsqueeze(2).repeat(1, 1, self.L) * self.coef.view(1, 1, self.L)
        s = torch.sin(rad).view(-1, dim * self.L)
        c = torch.cos(rad).view(-1, dim * self.L)

        return torch.stack([s, c], dim=2).view(-1, 2 * dim * self.L)


class SimpleEncoding(nn.Module):
    def __init__(self, L: int):
        super(SimpleEncoding, self).__init__()
        self.L = L

    def forward(self, vectors: T) -> T:
        elements = [vectors]
        for i in range(self.L):
            elements += [torch.sin(2.0**i * vectors), torch.cos(2.0**i * vectors)]
        return torch.concat(elements, dim=1)



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

            inside_ball_1 = torch.square(x-0.5) + torch.square(y) + torch.square(z) < 0.3
            inside_ball_2 = torch.square(x+0.5) + torch.square(y+0.2) + torch.square(z) < 0.8
            inside_balls = inside_ball_1 | inside_ball_2

            volume_density[inside_balls] = 1.0
            radiance[inside_balls, :] = torch.tensor([1.0, 0.0, 0.0])

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


class NeRF(nn.Module):  # Implementing Figure 7
    def __init__(self):
        super(NeRF, self).__init__()

        self.enc_position = PositionalEncoding(L=10)
        self.enc_direction = PositionalEncoding(L=4)

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

    def forward(self, position: T, direction: T) -> (T, T):
        p, d = self.enc_position(position), self.enc_direction(direction)

        x_1 = self.block_1(p)
        x_2 = self.block_2(torch.cat([x_1, p], dim=1))

        volume_density = torch.relu(x_2[:, 0])
        radiance = self.block_3(torch.cat([x_2[:, 1:(256+1)], d], dim=1))
  
        return volume_density, radiance


class TinyNeRF(nn.Module):
    def __init__(self): 
        super(TinyNeRF, self).__init__()
        
        input_dim = 60 + 24 # Positional plus direction encoding         
 
        self.enc_position = PositionalEncoding(L=10)
        self.enc_direction = PositionalEncoding(L=4)
 
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
        p, d = self.enc_position(position), self.enc_direction(direction)

        x = torch.concat([p, d], dim=1)
        y = torch.concat([self.block_1(x), x], dim=1)
        z = self.output_layer(self.block_2(y))
        return torch.relu(z[:, 3]), torch.sigmoid(z[:, :3])
