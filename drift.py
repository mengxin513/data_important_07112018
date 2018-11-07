
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
from scipy import ndimage
import matplotlib.pyplot as plt
from contextlib import contextmanager, closing
import data_file
import cv2
from camera_stuff import find_template
#import h5py
import threading
import queue
from PIL import Image

def image_capture(start_t, event, ms, q):
    while event.is_set():
        frame = ms.rgb_image().astype(np.float32)
        capture_t = time.time()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        q.put(frame)
        tim = capture_t - start_t
        q.put(tim)
        print('Number of itms in the queue: {}'.format(q.qsize()))
        time.sleep(0.2)

if __name__ == "__main__":

    with load_microscope("microscope_settings.npz") as ms, \
         closing(data_file.Datafile(filename = "drift.hdf5")) as df:

        assert picamera_supports_lens_shading(), "You need the updated picamera module with lens shading!"

        camera = ms.camera
        stage = ms.stage

        camera.resolution=(640,480)

        cam_pos = df.new_group("data", "drift")

        N_frames = 500
        #need to be consistant between drift.py and drift_plot.py

        camera.start_preview(resolution=(640,480))

        image = ms.rgb_image().astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean = np.mean(image)
        templ8 = (image - mean)[144:-144, 144:-144]
        cam_pos['template'] = templ8
        cam_pos['initial_image'] = image
        imgfile_location = "/home/pi/summer/drift/calibration/drift_templ8.jpg"
        cv2.imwrite(imgfile_location, templ8)
        imgfile_location = "/home/pi/summer/drift/calibration/drift_image.jpg"
        cv2.imwrite(imgfile_location, image)
        img = Image.open("/home/pi/summer/drift/calibration/drift_templ8.jpg")
        pad = Image.new('RGB', (352, 192)) #Tuple must be multiples of 32 and 16
        pad.paste(img, (0, 0))
        overlay = camera.add_overlay(pad.tobytes(), size = (352, 192))
        overlay.alpha = 128
        overlay.fullscreen = False
        overlay.layer = 3
        
        templ8_position = np.zeros((1, 2))
        frame = ms.rgb_image().astype(np.float32)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        templ8_position[0, :], corr = find_template(templ8, frame - np.mean(frame), return_corr = True, fraction=0.5)

        def move_overlay(cx, cy):
            """move the overlay to show a shift of cx,cy camera pixels"""
            x = int(960 + (cx - templ8_position[0, 0] - 176)*2.25)
            y = int(540 + (cy - templ8_position[0, 1] - 96)*2.25)
            overlay.window = (x, y, int(352*2.25), int(192*2.25))

        q = queue.Queue()
        event = threading.Event()

        start_t = time.time()
        t = threading.Thread(target = image_capture, args = (start_t, event, ms, q), name = 'thread1')
        event.set()
        t.start()

        try:
            while event.is_set():
                if not q.empty():
                    data = np.zeros((N_frames, 3))
                    for i in range(N_frames):
                        frame = q.get()
                        tim = q.get()
                        data[i, 0] = tim
                        data[i, 1:], corr = find_template(templ8, frame - np.mean(frame), return_corr = True, fraction=0.5)
                        move_overlay(*data[i, 1:3])
                    df.add_data(data, cam_pos, "data")
                    imgfile_location_1 = "/home/pi/summer/drift/frames/drift_%s.jpg" % time.strftime("%02Y.%02m.%02d_%02H:%02M:%02S")
                    imgfile_location_2 = "/home/pi/summer/drift/frames/corr_%s.jpg" % time.strftime("%02Y.%02m.%02d_%02H:%02M:%02S")
                    cv2.imwrite(imgfile_location_1, frame)
                    cv2.imwrite(imgfile_location_2, corr * 255.0 / np.max(corr))
                else:
                    time.sleep(0.5)
                print("Looping")
            print("Done")
        except KeyboardInterrupt:
            event.clear()
            t.join()
            camera.stop_preview()
            print ("Got a keyboard interrupt, stopping")
            