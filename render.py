import torch

T = torch.tensor

# From the original NeRF codebase, they arbitrarily set:
NEAR = 2.
FAR = 6.


def expected_colour(N: int, nerf: torch.nn, o: T, d: T, device: str = 'cpu'):  # Approx Eqn. 1
    batch_size = o.shape[0]

    # Eqn. 2 - Stratified sampling of points along ray
    t_n = torch.tensor([NEAR], device=device).repeat((batch_size, 1))
    t_f = torch.tensor([FAR], device=device).repeat((batch_size, 1))
    i = torch.arange(start=1, end=N + 1, step=1, device=device).reshape((1, N)).repeat(batch_size, 1)
    lower = t_n + (i - 1) / N * (t_f - t_n)
    upper = t_n + i / N * (t_f - t_n)

    # batch_size x N x 1
    t = torch.rand(batch_size, N, device=device) * (upper - lower) + lower

    positions = o.unsqueeze(1) + t.unsqueeze(2) * d.unsqueeze(1) # batch_size x 1 x 3 + batch_size x N x 1 * batch_size x 1 x 3
    positions = positions.view(batch_size*N, 3)  # batch_size*N x 3
    directions = d.repeat_interleave(N, dim=0)

    # Forward pass
    σ_i, c_i = nerf(positions, directions)
    σ_i = σ_i.view(batch_size, N)  # batch_size x N
    c_i = c_i.view(batch_size, N, 3)  # batch_size x N x 3

    # Eqn 3. - Quadrature
    δ_i = torch.cat([t[:, 1:] - t[:, :-1], torch.tensor(1e10).to(device).broadcast_to((batch_size, 1))], dim=1)
    α_dash = torch.exp(-σ_i*δ_i)
    α = 1 - α_dash 
    #T_i_inner = torch.cat([torch.tensor(1.0).to(device).broadcast_to((batch_size, 1)), α_dash[:, :-1] + 1e-10], -1)
    T_i_inner = α_dash + 1e-10 # confusing indices in formula provided, above should be correct but this works instead
    T_i = torch.cumprod(T_i_inner, -1)

    # Alpha compositing
    coefs = α * T_i 
    return torch.sum(coefs.unsqueeze(-1) * c_i, dim=1)