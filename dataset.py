import numpy as np

from torch.utils.data import Dataset
from tqdm import tqdm
from geometry import find_camera_rays, ray_cube_intersection
from load import View, load


def _preprocess_view(v: View):
    # Camera view to (image pixels, rays) array
    im, transform, camera_angle_x = v

    o, d = find_camera_rays(transform, camera_angle_x)
    im = im.reshape(800*800, 3) / 255.0

    t_intervals = [ray_cube_intersection(o[i, :], d[i, :]) for i in range(800*800)]
    t_intervals = np.array([[-np.inf, -np.inf] if t is None else t for t in t_intervals])

    return im.astype('float32'), o.astype('float32'), d.astype('float32'), t_intervals.astype('float32')


class NerfDataset(Dataset):
    def __init__(self, scene='chair', subset='train'):
        self.view_examples = []
        for v in tqdm(load(scene=scene, subset=subset), desc=f'Preprocess "{scene}:{subset}" '):
            self.view_examples += [_preprocess_view(v)]

        self.res = 800

    def __len__(self):
        return self.res * self.res * len(self.view_examples)

    def __getitem__(self, idx):
        view_idx, im_idx = divmod(idx, self.res * self.res)

        im, o, d, t_intervals = self.view_examples[view_idx]

        return im[im_idx, :], (o[im_idx, :], d[im_idx, :], t_intervals[im_idx, :])


if __name__ == '__main__':
    dataset = NerfDataset()
    print(dataset[2344])
