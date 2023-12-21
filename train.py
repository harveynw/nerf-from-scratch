import os
import torch
import wandb
import dataset

from torch.utils.data import DataLoader
from dataset import NerfDataset
from model import NeRF, TinyNeRF
from torch.optim.lr_scheduler import ExponentialLR
from run_model import compare_output
from loss import total_colour_loss, mean_colour_loss


# Configure dataset
train, val, test = NerfDataset('lego', 'train'), NerfDataset('lego', 'val'), NerfDataset('lego', 'test')
train_dataloader = DataLoader(train, batch_size=100, shuffle=True)
val_dataloader = DataLoader(val, batch_size=64, shuffle=True)

# Train
nerf: torch.nn.Module = NeRF()
# nerf: torch.nn.Module = TinyNeRF()

lr = 5e-4
eps = 1e-8
weight_decay = 0.5
enable_gradient_clip = False
gradient_clip = 1.0
n_epochs = 7

# Optimisation
loss = mean_colour_loss
optim = torch.optim.Adam(nerf.parameters(), lr=lr, eps=eps)
scheduler_test = ExponentialLR(optim, gamma=weight_decay)
scheduler = scheduler_test

batches_per_loss_report = 100
batches_per_checkpoint = 500
batches_per_gt_test = 500
batches_per_train_loop = 2000

PATH = 'model.pt'

wandb.init(
    project="nerf-from-scratch",
    config={
        "learning_rate": lr,
        "eps": eps,
        "enable_gradient_clip": enable_gradient_clip,
        "weight_decay": weight_decay,
        "gradient_clip": gradient_clip,
        "dataset": train.filename,
        "n_epochs": n_epochs,
    }
)

torch.set_default_dtype(torch.float32)
wandb.watch(nerf, log='all', log_freq=1)
device = os.getenv("DEVICE", "cpu")
nerf.to(device)


def train_loop(dataloader, model, loss_fn, optimiser, epoch):
    train_examples = batches_per_train_loop * dataloader.batch_size
    for batch, (rgb, rays) in enumerate(dataloader):
        # Backpropagation
        loss = loss_fn(model, rgb, rays, device=device).to('cpu')
        optimiser.zero_grad()
        loss.backward()

        # Step optimiser
        if enable_gradient_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimiser.step()
        wandb.log({"train_loss": loss})
        
        if batch % batches_per_checkpoint == 0 and batch > 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'loss': loss
            }, PATH)
        if batch % batches_per_gt_test == 0:
            fig, _ = compare_output(model, dataloader.dataset, view_idx=2, device=device)
            wandb.log({"comparison": fig}) 
        if batch % batches_per_loss_report == 0:
            loss, current = loss.item(), (batch + 1) * rgb.shape[0]
            print(f"loss: {loss:>7f}  [{current:>5d}/{train_examples:>5d}]")
        if batch == batches_per_train_loop-1:
            break

def val_loop(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    val_loss = 0

    with torch.no_grad():
        for rgb, rays in dataloader:
            val_loss += loss_fn(model, rgb, rays, device=device).to('cpu').item()

    val_loss /= num_batches

    wandb.log({"val_loss": val_loss})
    print(f"Validation Error: \n Avg loss: {val_loss:>8f} \n")


try:
    for t in range(n_epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, nerf, loss, optim, t + 1)
        val_loop(val_dataloader, nerf, loss)
        scheduler.step()
except KeyboardInterrupt:
    wandb.finish()
