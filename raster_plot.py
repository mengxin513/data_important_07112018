
import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages

def plotting1(x_data, y_data, x_label, y_label, group):
    fig, ax = plt.subplots(1, 1)
    ax.plot(x_data, y_data, "+")
    ax.set_aspect(1)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    r = max(xmax - xmin, ymax - ymin)
    ax.set_xlim((xmax + xmin)/2 - r/2, (xmax + xmin)/2 + r/2)
    ax.set_ylim((ymax + ymin)/2 - r/2, (ymax + ymin)/2 + r/2)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.suptitle(group.name[1:])

    plt.tight_layout()

    pdf.savefig(fig)
    plt.close(fig)

def plotting2(x_data, y_data, x_label, y_label, group):
    fig, ax = plt.subplots(1, 1)
    ax.plot(x_data, y_data, "+-")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.suptitle(group.name[1:])

    plt.tight_layout()

    pdf.savefig(fig)
    plt.close(fig)

if __name__ == "__main__":

    with PdfPages("raster_snake.pdf") as pdf:

        print ("Loading data...")

        df = h5py.File("raster.hdf5", mode = "r")
        for i in range(len(df)):
            group = df["raster%03d" % i]
            subgroup = group["snake_raster000"]
            n = len(subgroup)
            data = np.zeros([n, 6])
            for i in range(n):
                dset = subgroup["data%03d" % i]
                data[i, 0:3] = dset["cam_position"]
                data[i, 3:6] = dset["stage_position"]

            t = data[:, 0]
            pixel_shifts = np.zeros([n, 3])
            location_shifts = np.zeros([n, 3])

            for i in range(n):
                pixel_shifts[i, 0:2] = data[i, 1:3]
                pixel_shifts[i, 2] = 1
                location_shifts[i, 0] = data[i, 3]
                location_shifts[i, 1] = data[i, 5]
                location_shifts[i, 2] = 1
                
            # exclude points at the extreme of Y (the image analysis broke)
            # the first 3 points look dodgy, and we think they are at -14500
            # up to -12500, so let's exclude all points with stage y less
            # than -12000
            #stage_dy = location_shifts[:, 1]
            #stage_dy -= np.mean(stage_dy)
            #mask = stage_dy > -12000
            #masked_location_shifts = np.empty((int(np.sum(mask)), 3), dtype=np.float)
            #masked_pixel_shifts = np.empty_like(masked_location_shifts)
            #for i, j in enumerate(np.nonzero(mask)[0]):
            #    masked_location_shifts[i,:] = location_shifts[j,:]
            #    masked_pixel_shifts[i,:] = pixel_shifts[j,:]
            #location_shifts = masked_location_shifts
            #pixel_shifts = masked_pixel_shifts

            A, res, rank, s = np.linalg.lstsq(location_shifts, pixel_shifts)
            #A is the least squares solution pixcel_shifts*A = location_shifts
            #res is the sums of residuals location_shifts - pixcel_shifts*A
            #rank is rank of matrix pixcel_shifts
            #s is singular values of pixcel_shifts
            print(A)

            #unit vectors
            x = np.array([1, 0, 0]) 
            y = np.array([0, 1, 0])

            #dot products of A with x and y unit vectors to find x and y components of A
            A_x = np.dot(x, A) #the displacement in px corrosponding to 1 step in x
            A_y = np.dot(y, A)

            #uses standard dot product formula to find angle between A_x and A_y
            dotproduct = np.dot(A_x, A_y)
            cosa = dotproduct / (np.linalg.norm(A_x) * np.linalg.norm(A_y))
            angle = np.arccos(cosa)
            angle = angle * 180 / np.pi
            print (angle)

            transformed_stage_positions = np.dot(location_shifts, A)

            matplotlib.rcParams.update({'font.size': 12})

            plotting1(transformed_stage_positions[:, 0] - np.mean(transformed_stage_positions[:, 0]), transformed_stage_positions[:, 1] - np.mean(transformed_stage_positions[:, 1]), "Transformed Stage X Position [px]", "Transformed Stage Y Position [px]", group)
            for ia, na in enumerate(["X", "Y"]):
                plotting1(pixel_shifts[:, ia] - np.mean(pixel_shifts[:, ia]),
                          transformed_stage_positions[:, ia] - np.mean(transformed_stage_positions[:, ia]),
                          "Camera " + na + " Position [px]", "Transformed Stage " + na + " Position [px]", group)
            for ia, na in enumerate(["X", "Y"]):
                for ib, nb in enumerate(["X", "Y"]):
                    plotting2(pixel_shifts[:, ia] - np.mean(pixel_shifts[:, ia]),
                              transformed_stage_positions[:, ib] - pixel_shifts[:, ib],
                              "Camera " + na + " Position [px]", "Error in " + nb + " [px]", group)
            
    df.close()
