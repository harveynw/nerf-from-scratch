import torch
from torchviz import make_dot

from dataset import NerfDataset
from model import NeRF

# Render colour = Render(Forward pass(t_1) + Forward pass(t_2) + … )
#
# Optimised on (Render colour - g.t)

# this ensures that the current MacOS version is at least 12.3+
print(torch.backends.mps.is_available())
# this ensures that the current current PyTorch installation was built with MPS activated.
print(torch.backends.mps.is_built())

# Generate Visualisation from Forward Pass
nerf = NeRF()
dummy_x_1, dummy_x_2 = torch.ones((4096, 60)), torch.ones((4096, 24))
output_1, output_2 = nerf(dummy_x_1, dummy_x_2)
make_dot((output_1, output_2), params=dict(list(nerf.named_parameters()))).render("nerf_network", format="png")

