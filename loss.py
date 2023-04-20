import torch

from render import expected_colour


def total_colour_loss(model: torch.nn.Module, gt_color: torch.Tensor, rays: torch.Tensor, device: str = 'cpu') -> torch.tensor:
    o, d, t_n, t_f = rays[0], rays[1], rays[2][:, [0]], rays[2][:, [1]]
    o, d, t_n, t_f = o.to(device), d.to(device), t_n.to(device), t_f.to(device)
    gt_color = gt_color.to(device)

    c = expected_colour(N=64, nerf=model, o=o, d=d, t_n=t_n, t_f=t_f, device=device)
    loss = torch.sum(torch.sqrt(torch.sum(torch.square(gt_color - c), dim=1)))

    return loss


def mean_colour_loss(model: torch.nn.Module, gt_color: torch.Tensor, rays: torch.Tensor, device: str = 'cpu') -> torch.tensor:
    o, d, t_n, t_f = rays[0], rays[1], rays[2][:, [0]], rays[2][:, [1]]
    o, d, t_n, t_f = o.to(device), d.to(device), t_n.to(device), t_f.to(device)
    gt_color = gt_color.to(device)

    c = expected_colour(N=64, nerf=model, o=o, d=d, t_n=t_n, t_f=t_f, device=device)
    if torch.sum(torch.isnan(c)) > 0:
        print('nan forward pass on nerf model')
        exit()
        
    # loss = torch.sum(torch.sqrt(torch.sum(torch.square(gt_color - c), dim=1))) 
    loss = torch.mean(torch.square(gt_color - c))
    
    # batch_size = gt_color.shape[0]
    # return loss/batch_size
    return loss

