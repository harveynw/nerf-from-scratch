import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from dataset import NerfDataset
from model import NeRF, DebugNeRF
from render import expected_colour


def compare_output(model: torch.nn.Module, dataset: NerfDataset, view_idx: int = 0, device: str = 'mps'):
    # Compare output to g.t. on specified view
    v = dataset.view_examples[view_idx]

    o, d, t_n, t_f = torch.tensor(v[1]), torch.tensor(v[2]), torch.tensor(v[3][:, [0]]), torch.tensor(v[3][:, [1]])
    o, d, t_n, t_f = o.to(device), d.to(device), t_n.to(device), t_f.to(device)

    batch_size = 4096
    o_batches, d_batches, t_n_batches, t_f_batches = \
        list(torch.split(o, batch_size, 0)), \
            list(torch.split(d, batch_size, 0)), \
            list(torch.split(t_n, batch_size, 0)), \
            list(torch.split(t_f, batch_size, 0))

    model_outputs = []
    pbar = tqdm(total=len(o_batches))
    while len(o_batches) > 0:
        o_k, d_k, t_n_k, t_f_k = o_batches[0], d_batches[0], t_n_batches[0], t_f_batches[0]
        with torch.no_grad():
            model_outputs += [
                expected_colour(N=100, nerf=model, o=o_k, d=d_k, t_n=t_n_k, t_f=t_f_k, device=device).to('cpu')]
        o_batches.pop(0), d_batches.pop(0), t_n_batches.pop(0), t_f_batches.pop(0)
        pbar.update(1)
    pbar.close()

    model_output = torch.concat(model_outputs)
    generated_image = model_output.view(800, 800, 3)

    fig = plt.figure()
    ax_render = fig.add_subplot(121)
    ax_im = fig.add_subplot(122)

    ax_render.imshow(generated_image)
    ax_im.imshow(v[0].reshape(800, 800, 3))

    ax_render.set_title('NeRF')
    ax_im.set_title('Ground Truth')

    return fig, (ax_render, ax_im)


if __name__ == '__main__':
    # model = NeRF()
    # checkpoint = torch.load('model.pt')
    # model.load_state_dict(checkpoint['model_state_dict'])
    # model.eval()
    # model.to('mps')

    model = DebugNeRF('balls')

    dataset = NerfDataset('chair', 'train')

    fig, ax = compare_output(model, dataset, device='cpu')
    plt.show()


