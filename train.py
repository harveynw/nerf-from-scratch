import os
import pickle
import torch
import wandb

from torch.utils.data import DataLoader
from dataset import NerfDataset
from model import NeRF
from render import expected_colour_batched

# Load dataset, from .pickle or from fresh (takes a little while to process)
filename = 'dataset_save_chair.pickle'
if os.path.exists(filename):
    with open(filename, 'rb') as f:
        train, val, _ = pickle.load(f)
else:
    train, val, test = NerfDataset('chair', 'train'), NerfDataset('chair', 'val'), NerfDataset('chair', 'test')
    with open(filename, 'wb') as f:
        pickle.dump((train, val, test), f)

train_dataloader = DataLoader(train, batch_size=4096, shuffle=True)
val_dataloader = DataLoader(val, batch_size=256, shuffle=True)

# Train
lr = 5e-4
eps = 1e-7
weight_decay = 0.1

PATH = 'model.pt'

wandb.init(
    project="nerf-from-scratch",
    config={
        "learning_rate": lr,
        "eps": eps,
        "weight_decay": weight_decay,
        "dataset": filename,
        "epochs": 10,
    }
)

torch.set_default_dtype(torch.float32)
nerf: torch.nn.Module = NeRF()

mps_device = torch.device("mps")
nerf.to(mps_device)

optim = torch.optim.Adam(nerf.parameters(), lr=lr, eps=eps, weight_decay=weight_decay)


def train_loop(dataloader, model, loss_fn, optimiser, epoch):
    size = len(dataloader.dataset)
    for batch, (rgb, rays) in enumerate(dataloader):
        # Compute prediction and loss
        loss = loss_fn(model, rgb, rays, to_gpu=True)

        # Backpropagation
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        wandb.log({"train_loss": loss})
        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * rgb.shape[0]
            did_save = False

            if batch % 100 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    'loss': loss,
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


def loss_fn(model: torch.nn.Module, gt_color: torch.Tensor, rays: torch.Tensor, to_gpu: bool = False) -> torch.tensor:
    batch_size = gt_color.shape[0]
    o, d, t_n, t_f = rays[0], rays[1], rays[2][:, [0]], rays[2][:, [1]]

    drop = torch.isneginf(t_n).squeeze()
    gt_color, o, d, t_n, t_f = gt_color[~drop, :], o[~drop, :], d[~drop, :], t_n[~drop, :], t_f[~drop, :]

    device = 'cpu'
    if to_gpu:
        device = 'mps'
        o, d, t_n, t_f = o.to(device), d.to(device), t_n.to(device), t_f.to(device)

    c = expected_colour_batched(N=100, nerf=model, o=o, d=d, t_n=t_n, t_f=t_f, device=device)

    result = torch.sum(torch.sqrt(torch.sum(torch.square(gt_color - c), dim=1))) / batch_size  # done on CPU
    return result


try:
    n_epochs = 100

    for t in range(n_epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, nerf, loss_fn, optim, t + 1)
        val_loop(val_dataloader, nerf, loss_fn)

except KeyboardInterrupt:
    wandb.finish()
