import numpy as np
import matplotlib.pyplot as plt
from utils import DLT

def read_keypoints(filename):
    fin = open(filename, 'r')

    kpts = []
    while(True):
        line = fin.readline()
        if line == '': break

        line = line.split()
        line = [float(s) for s in line]

        line = np.reshape(line, (21, -1))
        kpts.append(line)

    kpts = np.array(kpts)
    return kpts


def visualize_3d(p3ds):

    """Apply coordinate rotations to point z axis as up"""
    Rz = np.array(([[0., -1., 0.],
                    [1.,  0., 0.],
                    [0.,  0., 1.]]))
    Rx = np.array(([[1.,  0.,  0.],
                    [0., -1.,  0.],
                    [0.,  0., -1.]]))

    p3ds_rotated = []
    for frame in p3ds:
        frame_kpts_rotated = []
        for kpt in frame:
            kpt_rotated = Rz @ Rx @ kpt
            frame_kpts_rotated.append(kpt_rotated)
        p3ds_rotated.append(frame_kpts_rotated)

    """this contains 3d points of each frame"""
    p3ds_rotated = np.array(p3ds_rotated)

    """Now visualize in 3D"""
    thumb_f = [[0,1],[1,2],[2,3],[3,4]]
    index_f = [[0,5],[5,6],[6,7],[7,8]]
    middle_f = [[0,9],[9,10],[10,11],[11, 12]]
    ring_f = [[0,13],[13,14],[14,15],[15,16]]
    pinkie_f = [[0,17],[17,18],[18,19],[19,20]]
    fingers = [pinkie_f, ring_f, middle_f, index_f, thumb_f]
    fingers_colors = ['red', 'blue', 'green', 'black', 'orange']

    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, kpts3d in enumerate(p3ds_rotated):
        if i%2 == 0: continue #skip every 2nd frame
        for finger, finger_color in zip(fingers, fingers_colors):
            for _c in finger:
                ax.plot(xs = [kpts3d[_c[0],0], kpts3d[_c[1],0]], ys = [kpts3d[_c[0],1], kpts3d[_c[1],1]], zs = [kpts3d[_c[0],2], kpts3d[_c[1],2]], linewidth = 4, c = finger_color)

        #draw axes
        ax.plot(xs = [0,5], ys = [0,0], zs = [0,0], linewidth = 2, color = 'red')
        ax.plot(xs = [0,0], ys = [0,5], zs = [0,0], linewidth = 2, color = 'blue')
        ax.plot(xs = [0,0], ys = [0,0], zs = [0,5], linewidth = 2, color = 'black')

        #ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.set_xlim3d(-7, 8)
        ax.set_xlabel('x')
        ax.set_ylim3d(-7, 8)
        ax.set_ylabel('y')
        ax.set_zlim3d(0, 15)
        ax.set_zlabel('z')
        ax.elev = 0.2*i
        ax.azim = 0.2*i
        plt.savefig('figs/fig_' + str(i) + '.png')
        plt.pause(0.01)
        ax.cla()


if __name__ == '__main__':

    p3ds = read_keypoints('kpts_3d.dat')
    visualize_3d(p3ds)
