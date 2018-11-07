
from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import h5py
import numpy.linalg
import matplotlib

if __name__ == "__main__":
    print ("Loading data...")

    microns_per_pixel = 2.16
    df = h5py.File("step_size.hdf5", mode = "r")
    group1 = df["data_steps000"]
    group2 = df["data_distance000"]
    dset1 = group1["data_steps00000"]
    dset2 = group2["data_distance00000"]

    matplotlib.rcParams.update({'font.size': 12})

    fig, ax1 = plt.subplots(1, 1)
    ax2 = ax1.twinx()

    ax1.plot(dset1[0:9], dset2[0:9] / dset1[0:9] * microns_per_pixel, "r-")
    ax2.plot(dset1[9:18], dset2[9:18] / dset1[9:18] * microns_per_pixel, "b-")

    ax1.set_xlabel('Steps Moved')
    ax1.set_ylabel('Step Size in X [$\mathrm{\mu m}$]')
    ax2.set_ylabel('Step Size in Y [$\mathrm{\mu m}$]')
    plt.savefig("step_size.pdf", bbox_inches='tight', dpi=180)

    df.close()

    plt.show()
