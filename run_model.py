import torch
import math
import matplotlib.pyplot as plt

from tqdm import tqdm
from dataset import NerfDataset
from model import NeRF, DebugNeRF, TinyNeRF
from render import expected_colour


def compare_output(model: torch.nn.Module, dataset: NerfDataset, view_idx: int = 0, device: str = 'mps', close_plt=True):
    # Compare output to g.t. on specified view
    v = dataset.view_examples[view_idx]

    o, d = torch.tensor(v.o), torch.tensor(v.d)
    o, d = o.to(device), d.to(device)

    batch_size = 4096
    batchify = lambda x: list(torch.split(x, batch_size, 0))

    o_batches, d_batches = batchify(o), batchify(d)

    model_outputs = []
    pbar = tqdm(total=len(o_batches))
    while len(o_batches) > 0:
        o_k, d_k = o_batches[0], d_batches[0]
        with torch.no_grad():
            model_outputs += [
                expected_colour(N=100, nerf=model, o=o_k, d=d_k, device=device).to('cpu')
            ]
        o_batches.pop(0), d_batches.pop(0)
        pbar.update(1)
    pbar.close()

    res = int(math.sqrt(dataset.get_pixels_per_image()))
    model_output = torch.concat(model_outputs)
    generated_image = model_output.view(res, res, 3)

    fig = plt.figure()
    ax_render = fig.add_subplot(121)
    ax_im = fig.add_subplot(122)

    ax_render.imshow(generated_image)
    ax_im.imshow(v.im.reshape(res, res, 3))

    ax_render.set_title('NeRF')
    ax_im.set_title('Ground Truth')
    if close_plt:
        plt.close()

    return fig, (ax_render, ax_im)


if __name__ == '__main__':
    #model = TinyNeRF()
    model = NeRF()
    checkpoint = torch.load('model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    #model = DebugNeRF('balls')
    #model = DebugNeRF('center_balls')
    dataset = NerfDataset('lego', 'test')

    for i, _ in enumerate(dataset.view_examples):
        fig, ax = compare_output(model, dataset, view_idx=i, device='cpu', close_plt=False)
        plt.show()


