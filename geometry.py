import numpy as np


def ray_cube_intersection(o, d):
    """
    Returns the values of t_n and t_f for the points of intersection of
    the given ray with the 3D cube centered at the origin and covering
    the interval [-1, 1] along each axis, or None if there is no
    intersection.

    Parameters:
    o (array-like): The origin of the ray.
    d (array-like): The direction of the ray.

    Returns:
    tuple or None: A tuple containing the values of t_n and t_f for the
        points of intersection of the ray with the cube, or None if there
        is no intersection.
    """

    # Compute the intervals of intersection of the ray with the planes
    # that define the cube.
    tmin = -np.inf
    tmax = np.inf
    for i in range(3):
        if d[i] == 0:
            if o[i] < -1 or o[i] > 1:
                return None
        else:
            t1 = (-1 - o[i]) / d[i]
            t2 = (1 - o[i]) / d[i]
            if t1 > t2:
                t1, t2 = t2, t1
            if t1 > tmin:
                tmin = t1
            if t2 < tmax:
                tmax = t2
            if tmin > tmax:
                return None

    # Return the interval of intersection if it exists.
    if tmin > 0:
        return tmin, tmax
    else:
        return None


def find_camera_rays(transform: np.ndarray, camera_angle_x: float) -> (np.ndarray, np.ndarray):
    H, W = 800, 800
    focal_length = .5 * W / np.tan(.5 * camera_angle_x)

    """ From the github code, Get ray origins, directions from a pinhole camera."""
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i - W * .5) / focal_length, -(j - H * .5) / focal_length, -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * transform[:3, :3], -1)
    rays_o = np.broadcast_to(transform[:3, -1], np.shape(rays_d))

    return rays_o.reshape(H*W, 3), rays_d.reshape(H*W, 3)
