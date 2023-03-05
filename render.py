import torch

T = torch.tensor


def _nan_check(x : T, name: str):
    count = torch.sum(torch.isnan(x)).to('cpu')
    if count > 0:
        print(f'ALERT check for NaN in {name} found {count}')
        exit()


def expected_colour(N: int, nerf: torch.nn, o: T, d: T, t_n: T, t_f: T, device: str = 'cpu'):  # Approx Eqn. 1
    batch_size = o.shape[0]

    # Eqn. 2 - Stratified sampling of points along ray
    i = torch.arange(start=1, end=N + 1, step=1, device=device).reshape((1, N)).repeat(batch_size, 1)
    lower = t_n + (i - 1) / N * (t_f - t_n)
    upper = t_n + i / N * (t_f - t_n)

    # batch_size x N x 1
    t = torch.rand(batch_size, N, device=device) * (upper - lower) + lower
    t = t.unsqueeze(2)

    o_t, d_t = o.unsqueeze(1), d.unsqueeze(1)  # batch_size x 1 x 3
    positions = o_t + t * d_t
    positions = positions.view(batch_size*N, 3)  # batch_size*N x 3
    directions = d.repeat_interleave(N, dim=0)

    _nan_check(positions, 'Positions'), _nan_check(directions, 'Directions')

    σ_i, c_i = nerf(positions, directions)

    _nan_check(σ_i, 'NN Output Volume Density')
    _nan_check(c_i, 'NN Output Radiance')

    σ_i = σ_i.view(batch_size, N)  # batch_size x N
    c_i = c_i.view(batch_size, N, 3)  # batch_size x N x 3

    # Eqn 3. - Quadrature
    t = t.squeeze()  # batch_size x N
    δ_i = torch.cat([t[:, 1:], t_f], dim=1) - t  # batch_size x N, δ_N := t_f - t_N

    T_i_inner = -σ_i*δ_i
    T_i_inner = torch.cat([torch.zeros(batch_size, 1, device=device), T_i_inner[:, :-1]], dim=1)
    T_i = torch.exp(torch.cumsum(T_i_inner, dim=1))

    _nan_check(T_i, 'T_i')

    coef_term = 1 - torch.exp(-σ_i*δ_i)
    coef = T_i * coef_term
    C = coef.unsqueeze(dim=2) * c_i
    return C.sum(dim=1)


if __name__ == '__main__':
    from model import NeRF

    nerf = NeRF()

    N = 10000
    o = [torch.tensor([0, 0, 0]), torch.tensor([5, 0, 6]), torch.tensor([8, 0, 2])]
    d = [torch.tensor([3, 2, 5]), torch.tensor([8, 9, 1]), torch.tensor([5, 5, 5])]
    t_n = [torch.tensor([0.1]), torch.tensor([0.4]), torch.tensor([0.5])]
    t_f = [torch.tensor([0.9]), torch.tensor([0.5]), torch.tensor([0.8])]

    print(expected_colour(N, nerf, torch.stack(o), torch.stack(d), torch.stack(t_n), torch.stack(t_f)))

