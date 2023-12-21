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
