import torch

from render import expected_colour


def total_colour_loss(model: torch.nn.Module, gt_color: torch.Tensor, rays: torch.Tensor, device: str = 'cpu') -> torch.tensor:
    o, d = rays[0].to(device), rays[1].to(device)
    gt_color = gt_color.to(device)

    c = expected_colour(N=64, nerf=model, o=o, d=d, device=device)
    loss = torch.sum(torch.sqrt(torch.sum(torch.square(gt_color - c), dim=1)))
    return loss


def mean_colour_loss(model: torch.nn.Module, gt_color: torch.Tensor, rays: torch.Tensor, device: str = 'cpu') -> torch.tensor:
    o, d = rays[0].to(device), rays[1].to(device)
    gt_color = gt_color.to(device)

    c = expected_colour(N=64, nerf=model, o=o, d=d, device=device)
    loss = torch.mean(torch.square(gt_color - c))
    return loss
