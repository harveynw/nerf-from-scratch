import os
import torch
import wandb

from torch.utils.data import DataLoader
from dataset import NerfDataset
from model import NeRF
from render import expected_colour
from torch.optim.lr_scheduler import ExponentialLR

from run_model import compare_output

# Load dataset, from .pickle or from fresh (takes a little while to process)
train, val, test = NerfDataset('chair', 'train'), NerfDataset('chair', 'val'), NerfDataset('chair', 'test')

train_dataloader = DataLoader(train, batch_size=4096, shuffle=True)
val_dataloader = DataLoader(val, batch_size=256, shuffle=True)

# Train
# lr = 5e-4
lr = 5e-6
# eps = 1e-7
weight_decay = 0.1
gradient_clip = 1e-2
n_epochs = 7

PATH = 'model.pt'

wandb.init(
    project="nerf-from-scratch",
    config={
        "learning_rate": lr,
        # "eps": eps,
        "weight_decay": weight_decay,
        "gradient_clip": gradient_clip,
        "dataset": train.filename,
        "n_epochs": n_epochs,
    }
)

torch.set_default_dtype(torch.float32)
nerf: torch.nn.Module = NeRF()
device = os.getenv("DEVICE", "cpu")
nerf.to(device)

# optim = torch.optim.Adam(nerf.parameters(), lr=lr, eps=eps)
optim = torch.optim.Adam(nerf.parameters(), lr=lr)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim, gamma=weight_decay)
scheduler_test = ExponentialLR(optim, gamma=0.9)
scheduler = scheduler_test


def train_loop(dataloader, model, loss_fn, optimiser, epoch):
    size = len(dataloader.dataset)
    for batch, (rgb, rays) in enumerate(dataloader):

        # with torch.autograd.set_detect_anomaly(True):
        # Compute prediction and loss
        loss = loss_fn(model, rgb, rays, device=device).to('cpu')

        # Backpropagation
        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimiser.step()

        # Debugging
        # nan_count = 0
        # nan_count += torch.sum(torch.isnan(rgb)).item()
        # nan_count += torch.sum(torch.isnan(rays[0])).item()
        # nan_count += torch.sum(torch.isnan(rays[1])).item()
        # nan_count += torch.sum(torch.isnan(rays[2])).item()
        nan_gradients = sum([torch.sum(torch.isnan(p.grad)).item() for p in model.parameters()])
        # print(f'Batch Size: {rgb.shape[0]} NaN Count Dataset: {nan_count} NaN Gradients: {nan_gradients}')

        if nan_gradients > 0:
            print('nan gradient')
            exit()

        wandb.log({"train_loss": loss})
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * rgb.shape[0]
            did_save = False

            if batch % 4 == 0:
                fig, _ = compare_output(model, dataloader.dataset, device=device)
                wandb.log({"comparison": fig})

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    'loss': loss
                }, PATH)
                did_save = True

            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]", "(Checkpoint!)" if did_save else "")


def val_loop(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    val_loss, correct = 0, 0

    with torch.no_grad():
        for rgb, rays in dataloader:
            val_loss += loss_fn(model, rgb, rays).item()

    val_loss /= num_batches

    wandb.log({"val_loss": val_loss})
    print(f"Validation Error: \n Avg loss: {val_loss:>8f} \n")


def loss_fn(model: torch.nn.Module, gt_color: torch.Tensor, rays: torch.Tensor, device: str = 'cpu') -> torch.tensor:
    batch_size = gt_color.shape[0]
    o, d, t_n, t_f = rays[0], rays[1], rays[2][:, [0]], rays[2][:, [1]]
    o, d, t_n, t_f = o.to(device), d.to(device), t_n.to(device), t_f.to(device)
    gt_color = gt_color.to(device)

    c = expected_colour(N=100, nerf=model, o=o, d=d, t_n=t_n, t_f=t_f, device=device)
    loss = torch.sum(torch.sqrt(torch.sum(torch.square(gt_color - c), dim=1))) / batch_size

    return loss


try:
    for t in range(n_epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, nerf, loss_fn, optim, t + 1)

        scheduler.step()

        val_loop(val_dataloader, nerf, loss_fn)

except KeyboardInterrupt:
    wandb.finish()
