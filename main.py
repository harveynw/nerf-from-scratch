import torch

# Render colour = Render(Forward pass(t_1) + Forward pass(t_2) + … )
#
# Optimised on (Render colour - g.t)

# this ensures that the current MacOS version is at least 12.3+
print(torch.backends.mps.is_available())
# this ensures that the current current PyTorch installation was built with MPS activated.
print(torch.backends.mps.is_built())
