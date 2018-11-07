
from __future__ import print_function
import io
import sys
import os
import time
import argparse
import numpy as np
import picamera
from builtins import input
from readchar import readchar, readkey
from openflexure_stage import OpenFlexureStage
from openflexure_microscope import load_microscope
from openflexure_microscope.microscope import picamera_supports_lens_shading
import scipy
from scipy import ndimage, signal
import matplotlib.pyplot as plt
from contextlib import contextmanager, closing
import data_file
import cv2
from camera_stuff import find_template
#import h5py
import threading
import queue

def movement(step, event, ms):
    while not event.wait(1):
            ms.stage.move_rel(step)

def printProgressBar(iteration, total, length = 10):
    percent = 100.0 * iteration / total
    filledLength = int(length * iteration // total)
    bar = '*' * filledLength + '-' * (length - filledLength)
    print('Progress: |%s| %d%% Completed' % (bar, percent), end = '\r')
    if iteration == total: 
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Finds the smallest resolvable step")
    parser.add_argument("step", type = int, nargs = 3, help = "The displacement between each point, in steps")
    parser.add_argument("--n_frames", type = int, default = 2000, help = "The number of frames to record for each run")
    parser.add_argument("--framerate", type=int, default= 100, help = "Rate at which to run the camera (frames/second)")
    args = parser.parse_args()

    with load_microscope("microscope_settings.npz", dummy_stage = False) as ms, \
         closing(data_file.Datafile(filename = "precision.hdf5")) as df:

        assert picamera_supports_lens_shading(), "You need the updated picamera module with lens shading!"

        camera = ms.camera
        stage = ms.stage

        N_frames = args.n_frames
        step = args.step
        framerate = args.framerate
        backlash = 256

        camera.resolution=(640,480)
        camera.framerate = framerate
        stage.backlash = backlash

        cam_pos = df.new_group("data", "precision")
        data = np.zeros((N_frames, 3))

        outputs = [io.BytesIO() for i in range(N_frames)]

        camera.start_preview(resolution=(640,480))

        stage.move_rel([-backlash, -backlash, -backlash])
        stage.move_rel([backlash, backlash, backlash])

        image = ms.rgb_image().astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean = np.mean(image)
        templ8 = (image - mean)[100:-100, 100:-100]
        imgfile_location = "/home/pi/summer/drift/calibration/precision_templ8.jpg"
        cv2.imwrite(imgfile_location, templ8)
        imgfile_location = "/home/pi/summer/drift/calibration/precision_image.jpg"
        cv2.imwrite(imgfile_location, image)

        event = threading.Event()

        initial_stage_position = stage.position
        t = threading.Thread(target = movement, args = (step, event, ms), name = 'thread1')
        t.start()

        try:
            start_t = time.time()
            camera.capture_sequence(outputs, 'jpeg', use_video_port=True)
            end_t = time.time()
        finally:
            event.set()
            t.join()
            stage.move_abs(initial_stage_position)
            camera.stop_preview()
            print ("Stopping...")

        print("Recorded {} frames in {} seconds ({} fps)".format(N_frames, end_t - start_t, N_frames / (end_t - start_t)))
        print("Camera framerate was set to {}, and reports as {}".format(framerate, camera.framerate))

        for j, k in enumerate(outputs):
                frame_data = np.fromstring(k.getvalue(), dtype = np.uint8)
                frame = cv2.imdecode(frame_data, 1)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                data[j, 1:], corr = find_template(templ8, frame - np.mean(frame), return_corr = True, fraction = 0.5)
                data[j, 0] = float(j) / float(framerate)
                printProgressBar(j, N_frames)
        print("")

        df.add_data(data, cam_pos, "data")
