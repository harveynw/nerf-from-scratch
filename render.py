import torch

from model import gamma_encoding

T = torch.tensor


def expected_colour(N: int, nerf: torch.nn, o: T, d: T, t_n: T, t_f: T):  # Approx Eqn. 1
    # Eqn. 2 - Stratified sampling of points along ray
    i = torch.arange(start=1, end=N + 1, step=1)
    upper = t_n + (i - 1) / N * (t_f - t_n)
    lower = t_n + i / N * (t_f - t_n)
    t = torch.rand(N) * (upper - lower) + lower

    # Forward pass on network, from paper: L = 10 for γ(x) and L = 4 for γ(d) for position and view direction
    position_encoded = torch.stack([gamma_encoding(o + t_i * d, 10) for t_i in t])
    direction_encoded = gamma_encoding(d, 4).repeat(N, 1)
    σ_i, c_i = nerf(position_encoded, direction_encoded)

    # Eqn 3. - Quadrature
    δ_i = torch.cat([t[1:], t_f.reshape((1,))]) - t  # Used δ_N := t_f - t_N
    T_i = torch.tensor([torch.exp(
        -torch.sum(torch.tensor([σ_i[j] * δ_i[j] for j in range(ii-1)]))
    ) for ii in i])
    C = torch.matmul(T_i * (1 - torch.exp(-σ_i * δ_i)), c_i)

    return C


if __name__ == '__main__':
    from model import NeRF

    nerf = NeRF()
    print(expected_colour(N=100,
                          nerf=nerf,
                          o=torch.tensor([0, 0, 0]),
                          d=torch.tensor([3.0, 2.0, 5.0]),
                          t_n=torch.tensor([0.1]),
                          t_f=torch.tensor([0.9])))
