import numpy as np
import matplotlib.pyplot as plt
from math import pi


def draw_2d_scatter(name, x, y, xlabel, ylabel):
    plt.scatter(x, y, s=1)
    xlim = 1.1 * np.max(np.abs(x))
    ylim = 1.1 * np.max(np.abs(y))
    ax = plt.gca()
    ax.set_xlim([-xlim, xlim])
    ax.set_ylim([-ylim, ylim])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(name)
    plt.close()


def draw_3d_scatter(name, x, y, z, xlabel, ylabel, zlabel):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z, s=1)
    xlim = 1.1 * np.max(np.abs(x))
    ylim = 1.1 * np.max(np.abs(y))
    zlim = 1.1 * np.max(np.abs(z))
    ax.set_xlim([-xlim, xlim])
    ax.set_ylim([-ylim, ylim])
    ax.set_zlim([-zlim, zlim])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.savefig(name)


object_pose = np.load('object_pose.npy').item()
delta_object_pose = object_pose['end'] - object_pose['start']
dx = delta_object_pose[:, 1]
dy = delta_object_pose[:, 2]
dtheta = delta_object_pose[:, 3]
dtheta[dtheta > pi] -= 2 * pi
dtheta[dtheta < -pi] += 2 * pi


draw_2d_scatter('object_pose_xy.png', dx, dy,
                r'$\Delta x$', r'$\Delta y$')
draw_2d_scatter('object_pose_xtheta.png', dx, dtheta,
                r'$\Delta x$', r'$\Delta \theta$')
draw_2d_scatter('object_pose_ytheta.png', dy, dtheta,
                r'$\Delta y$', r'$\Delta \theta$')


draw_3d_scatter('object_pose_xytheta.png', dx, dy, dtheta,
                r'$\Delta x$', r'$\Delta y$', r'$\Delta \theta$')
