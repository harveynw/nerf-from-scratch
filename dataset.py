import numpy as np

from torch.utils.data import Dataset
from tqdm import tqdm
from geometry import find_camera_rays, ray_cube_intersection
from load import View, load


def _preprocess_view(v: View, filter_empty_rays: bool):
    # Camera view to (image pixels, rays) array
    im, transform, camera_angle_x = v

    o, d = find_camera_rays(transform, camera_angle_x)
    im = im.reshape(800*800, 3) / 255.0

    t_intervals = [ray_cube_intersection(o[i, :], d[i, :]) for i in range(800*800)]
    t_intervals = np.array([[-np.inf, -np.inf] if t is None else t for t in t_intervals])

    if filter_empty_rays:
        drop = ~np.isneginf(t_intervals[:, 0])
        im, o, d, t_intervals = im[drop, :], o[drop, :], d[drop, :], t_intervals[drop, :]

    return im.astype('float32'), o.astype('float32'), d.astype('float32'), t_intervals.astype('float32')


class NerfDataset(Dataset):
    def __init__(self, scene='chair', subset='train', max_views=-1, filter_empty_rays=True):
        self.view_examples = []
        view_count = 0
        for v in tqdm(load(scene=scene, subset=subset), desc=f'Preprocess "{scene}:{subset}" '):
            self.view_examples += [_preprocess_view(v, filter_empty_rays)]

            view_count += 1
            if view_count == max_views:
                break

        self.res = 800

    def __len__(self):
        return sum([v[0].shape[0] for v in self.view_examples])

    def __getitem__(self, idx):
        for v in self.view_examples:
            view_examples = v[0].shape[0]
            if idx > view_examples:
                idx -= view_examples
            else:
                im, o, d, t_intervals = v
                return im[idx, :], (o[idx, :], d[idx, :], t_intervals[idx, :])

        return None


if __name__ == '__main__':
    dataset = NerfDataset()
    print(dataset[2344])
