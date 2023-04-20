import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from geometry import find_camera_rays
from load import load


def arrow_3d(ax: plt.axis, origin, dir, **kwargs):
    """ Arrow 3D

    Plots a 3D arrow on ax
    """
    u, v, w = [dir[0]], [dir[1]], [dir[2]]
    ax.quiver([origin[0]], [origin[1]], [origin[2]], u, v, w, **kwargs)


def point_3d(ax: plt.axis, point, **kwargs):
    """ Point 3D

    Plots a single point on ax
    """
    ax.scatter([point[0]], [point[1]], [point[2]], **kwargs)


def points_3d(ax: plt.axis, points: np.ndarray, **kwargs):
    """ Points 3D

    Plots 3D points on ax
    """
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], **kwargs)


def plot_unit_cube(ax):
    # create cube data
    top_verts = [(1, 1, 1), (-1, 1, 1), (-1, -1, 1), (1, -1, 1)]
    bottom_verts = [(1, 1, -1), (-1, 1, -1), (-1, -1, -1), (1, -1, -1)]
    side_verts = [(1, 1, 1), (-1, 1, 1), (-1, 1, -1), (1, 1, -1), (1, 1, 1),
                  (1, -1, 1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1), (-1, -1, -1),
                  (1, -1, -1), (1, -1, 1), (-1, -1, 1), (-1, -1, -1), (1, -1, -1),
                  (1, 1, -1), (-1, 1, -1), (-1, -1, -1), (1, -1, -1), (1, 1, -1)]

    # create Poly3DCollection objects for cube faces
    top_face = Poly3DCollection([top_verts], alpha=0.25, facecolor='gray', edgecolor='black')
    bottom_face = Poly3DCollection([bottom_verts], alpha=0.25, facecolor='gray', edgecolor='black')
    side_face = Poly3DCollection([side_verts], alpha=0.25, facecolor='gray', edgecolor='black')

    # add faces to axis
    ax.add_collection3d(top_face)
    ax.add_collection3d(bottom_face)
    ax.add_collection3d(side_face)


def plot_rays(ax, transform, camera_angle_x):
    origins, directions = find_camera_rays(transform, camera_angle_x)

    dirs = origins + directions

    selection = np.random.choice(origins.shape[0], 1000, replace=False)

    points_3d(ax, origins[selection, :], s=1)
    points_3d(ax, dirs[selection, :], s=1)

    # Select random ray
    idx = np.random.randint(low=0, high=origins.shape[0])
    t_linspace = np.linspace(start=2., stop=6., num=100).reshape((-1, 1))
    o, d = origins[idx, :].reshape((1, 3)), directions[idx, :].reshape((1, 3))
    points_3d(ax, o + t_linspace * d, s=1)


for item in load('chair', 'train'):
    im, transform, camera_angle_x = item

    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax_im = fig.add_subplot(122)
    ax_im.imshow(im)

    lim = 5
    ax.set_xlim((-lim, lim)), ax.set_ylim((-lim, lim)), ax.set_zlim((-lim, lim))

    plot_rays(ax, transform, camera_angle_x)
    plot_unit_cube(ax)

    plt.show()
