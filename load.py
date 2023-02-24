import json
import os
import gdown
import numpy as np

from os import path
from PIL import Image
from collections import namedtuple


message = """
    Could not find the NeRF Synthetic dataset.
    You can do this manually (fastest!) by following this:
    
        Create a folder in root called "data/". Then navigate to the NeRF paper
        site (https://www.matthewtancik.com/nerf), click "Data". Download 
        nerf_synthetic and unzip in the data folder.
        
    Otherwise this script will download it for you, just press enter.
"""

if not path.exists("data/nerf_synthetic"):
    print(message)
    _ = input()

    if not path.exists("data"):
        os.mkdir("data")

    # # Download
    url = "https://drive.google.com/uc?id=18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG"
    output = "data/nerf_synthetic.zip"
    gdown.download(url=url, output=output, quiet=False)

    # Unzip
    import zipfile
    with zipfile.ZipFile(output, "r") as zip_ref:
        zip_ref.extractall("data")

View = namedtuple('View', 'im transform camera_angle_x')


def _get_json_file(dir: str, subset: str):
    with open(dir + f'/transforms_{subset}.json') as json_file:
        json_dict = json.load(json_file)
    return json_dict


def _parse_frame(dir: str, frame_dict: dict):
    file_path = frame_dict['file_path']
    transform = np.array(frame_dict["transform_matrix"])

    rgba_im = Image.open(path.join(dir, file_path) + '.png')
    rgba_im.load()
    im = Image.new("RGB", rgba_im.size, (255, 255, 255))
    im.paste(rgba_im, mask=rgba_im.split()[3])

    return np.array(im), transform


def _parse_frames(dir: str, json_dict: dict):
    frames_dict = json_dict['frames']

    parsed_frames = []
    for frame_dict in frames_dict:
        im, transform = _parse_frame(dir, frame_dict)
        parsed_frames += [View(im, transform, json_dict['camera_angle_x'])]

    return parsed_frames


def load(scene: str = 'chair', subset: str = 'train'):
    dir = path.join("data/nerf_synthetic", scene)

    assert path.exists(dir), f"Could not find a dataset for '{scene}'"
    assert path.exists(path.join(dir, subset)), f"Could not find a dataset for '{scene}:{subset}'"

    json_dict = _get_json_file(dir, subset)

    return _parse_frames(dir, json_dict)
