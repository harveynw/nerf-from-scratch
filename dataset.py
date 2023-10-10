import random
import numpy as np
import pickle
import os

from typing import List
from dataclasses import dataclass
from torch.utils.data import Dataset, IterableDataset
from tqdm import tqdm
from geometry import find_camera_rays
from load import View, load

# From the original NeRF codebase, they arbitrarily set:
NEAR = 2.
FAR = 6.


@dataclass
class View:
    im: np.ndarray
    o: np.ndarray
    d: np.ndarray


def _preprocess_view(v: View):
    # Camera view to (image pixels, rays) data structure
    im, transform, camera_angle_x = v
    o, d = find_camera_rays(transform, camera_angle_x, im.shape[0], im.shape[1])
    im, o, d = im.astype('float32'), o.astype('float32'), d.astype('float32')

    n_pixels = im.shape[0]*im.shape[1]

    return View(im=im.reshape(n_pixels, 3) / 255.0, o=o, d=d)


class NerfDataset(Dataset):
    def __init__(self, scene='chair', subset='train', max_views=-1, refresh=False):
        self.id = f'{scene}_{subset}_{"all" if max_views == -1 else max_views}'
        self.filename = self.id + '.pickle'
        self.cache_path = os.path.join('cache', self.filename)

        if os.path.exists(self.cache_path) and not refresh:
            with open(self.cache_path, 'rb') as f:
                self.__dict__ = pickle.load(f)
        else:
            self.view_examples: List[View] = []
            view_count = 0
            for v in tqdm(load(scene=scene, subset=subset), desc=f'Preprocess "{scene}:{subset}" '):
                self.view_examples += [_preprocess_view(v)]
                view_count += 1
                if view_count == max_views:
                    break

            self.pixels_per_image = self.view_examples[0].im.shape[0]

            if not os.path.exists('cache'):
                os.mkdir('cache')
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self.__dict__, f)

    def __len__(self):
        return self.pixels_per_image * len(self.view_examples)

    def __getitem__(self, idx):
        view_idx = idx // self.pixels_per_image
        idx = idx % self.pixels_per_image

        v = self.view_examples[view_idx]
        return v.im[idx, :], (v.o[idx, :], v.d[idx, :])

    def get_pixels_per_image(self):
        return self.pixels_per_image


#class NerfDatasetGrouped(IterableDataset):
#    # NerfDataset but an iterable-style dataset where shuffling is applied within views and not as a whole
#    def __init__(self, scene='chair', subset='train', max_views=-1, refresh=False):
#        self.id = f'{scene}_{subset}_{"all" if max_views == -1 else max_views}'
#        self.filename = self.id + '.pickle'
#        self.cache_path = os.path.join('cache', self.filename)
#
#        if os.path.exists(self.cache_path) and not refresh:
#            with open(self.cache_path, 'rb') as f:
#                self.__dict__ = pickle.load(f)
#        else:
#            self.view_examples = []
#
#            view_count = 0
#            for v in tqdm(load(scene=scene, subset=subset), desc=f'Preprocess "{scene}:{subset}" '):
#                self.res = (v[0].shape[0], v[0].shape[1])
#                self.view_examples += [_preprocess_view(v)]
#
#                view_count += 1
#                if view_count == max_views:
#                    break
#
#            self.n_pixels = self.res[0] * self.res[1]
#            print('Set n_pixels as', self.n_pixels)
#
#            if not os.path.exists('cache'):
#                os.mkdir('cache')
#            with open(self.cache_path, 'wb') as f:
#                pickle.dump(self.__dict__, f)
#
#    def _reset_pixel_order(self):
#        order = np.arange(self.n_pixels)
#        np.random.shuffle(order)
#        self.pixel_order = order
#
#    def __iter__(self):
#        # Reset to first view
#        print('n_pixels iters', self.n_pixels)
#        self.view_cursor = 0
#        self.pixel_cursor = -1
#        self._reset_pixel_order()
#        return self
#
#    def __next__(self):
#        # Return new pixel
#        self.pixel_cursor += 1
#
#        if self.pixel_cursor == self.n_pixels:
#            self.view_cursor += 1
#            self.pixel_cursor = -1
#            self._reset_pixel_order()
#
#        if self.view_cursor == len(self.view_examples):
#            raise StopIteration
#
#        im, o, d, t_intervals = self.view_examples[self.view_cursor]
#        idx = self.pixel_order[self.pixel_cursor]
#
#        return im[idx, :], (o[idx, :], d[idx, :], t_intervals[idx, :])
#
#    def __len__(self):
#        return self.n_pixels * len(self.view_examples)
#
#    def get_res(self):
#        return self.res


if __name__ == '__main__':
    dataset = NerfDataset()
    for _ in range(100):
        print(dataset[int(random.random() * len(dataset))])
