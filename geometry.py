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
    # Camera intrinsic matrix
    focal_length = 0.5 * 800 / np.tan(0.5 * camera_angle_x)
    K = np.array([[focal_length, 0, 400], [0, focal_length, 400], [0, 0, 1]])

    # Ray origins
    C = transform[:3, 3]

    # Pixel coordinates
    y, x = np.mgrid[:800, :800]
    pixels = np.column_stack((x.ravel(), y.ravel(), np.ones_like(x.ravel())))

    # Ray directions
    R = np.linalg.inv(K) @ pixels.T
    R = transform[:3, :3] @ R
    R /= np.linalg.norm(R, axis=0)

    # Ray origins and directions
    origins = np.tile(C, (R.shape[1], 1)).T
    directions = R

    return origins.reshape((3, 800*800)).T, -directions.reshape((3, 800*800)).T
