import numpy as np


def find_camera_rays(transform: np.ndarray, camera_angle_x: float, height: int, width: int) -> (np.ndarray, np.ndarray):
    focal_length = .5 * width / np.tan(.5 * camera_angle_x)

    """ From the original github repo, Get ray origins, directions from a pinhole camera."""
    i, j = np.meshgrid(np.arange(width, dtype=np.float32),
                       np.arange(height, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i - width * .5) / focal_length, -(j - height * .5) / focal_length, -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * transform[:3, :3], -1)
    rays_o = np.broadcast_to(transform[:3, -1], np.shape(rays_d))

    # We ensure d are unit vectors
    rays_d = rays_d.reshape(height*width, 3)
    rays_d = rays_d / np.expand_dims(np.linalg.norm(rays_d, axis=1), axis=1)

    return rays_o.reshape(height*width, 3), rays_d
