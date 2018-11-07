
from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib

def printProgressBar(iteration, total, length = 10):
    percent = 100.0 * iteration / total
    filledLength = int(length * iteration // total)
    bar = '*' * filledLength + '-' * (length - filledLength)
    print('Progress: |%s| %d%% Completed' % (bar, percent), end = '\r')
    if iteration == total: 
        print()

if __name__ == "__main__":
    print ("Loading data...")

    N_frames = 500
    #need to be consistant between drift.py and drift_plot.py

    df = h5py.File("drift.hdf5", mode = "r")
    group = df.values()[-1]
    N_points = len(group) - 2
    data = np.zeros([N_frames * N_points, 3])
    for i in range(N_points):
        dset = group["data%05d" % i]
        for j in range(N_frames):
            data[N_frames * i + j, :] = dset[j, :]
        printProgressBar(i, N_points)
    print("")

    matplotlib.rcParams.update({'font.size': 12})

    microns_per_pixel = 2.16

    t = data[:, 0]
    x = data[:, 1] * microns_per_pixel
    x -= np.mean(x)
    y = data[:, 2] * microns_per_pixel
    y -= np.mean(y)

    fig, ax = plt.subplots(1, 1)

    ax.plot(t, x, "r-")
    ax2 = ax.twinx()
    ax2.plot(t, y, "b-")

    # Make the scale the same for X and Y (so it's obvious which is moving)
    xmin, xmax = ax.get_ylim()
    ymin, ymax = ax2.get_ylim()
    r = max(xmax - xmin, ymax - ymin)
    ax.set_ylim((xmax + xmin)/2 - r/2, (xmax + xmin)/2 + r/2)
    ax2.set_ylim((ymax + ymin)/2 - r/2, (ymax + ymin)/2 + r/2)

    ax.set_xlabel('Time [$\mathrm{s}$]')
    ax.set_ylabel('X Position [$\mathrm{\mu m}$]')
    ax2.set_ylabel('Y Position [$\mathrm{\mu m}$]')

    fig.suptitle(group.name[1:])

    plt.savefig("drift.pdf", bbox_inches='tight', dpi=180)

    df.close()

    plt.show()
