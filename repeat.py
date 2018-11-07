
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
import random
from PIL import Image

def measure_txy(start_t, ms, templ8):
    data = np.zeros((1, 3))
    data[0, 0] = time.time() - start_t
    frame = ms.rgb_image().astype(np.float32)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    data[0, 1:], corr = find_template(templ8, frame - np.mean(frame), return_corr = True, fraction=0.5)
    camera.stop_preview()
    cv2.imshow("corr", corr * 255.0 / np.max(corr))
    cv2.waitKey(1000)
    camera.start_preview()
    return data

def random_point(move_dist):
    angle = random.randrange(0, 360) * np.pi / 180
    vector = np.array([move_dist * np.cos(angle), 0, move_dist * np.sin(angle)])
    vector = np.rint(vector)
    return vector

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Characterises repeatability of the instrument")
    parser.add_argument("n_moves", type=int, help="Number of randoms moves for each displacement")
    parser.add_argument("--n_displacement", type=int, default=10, help="Number of different displacements")
    args = parser.parse_args()

    with load_microscope("microscope_settings.npz", dummy_stage = False) as ms, \
         closing(data_file.Datafile(filename = "repeat.hdf5")) as df:

        assert picamera_supports_lens_shading(), "You need the updated picamera module with lens shading!"

        camera = ms.camera
        stage = ms.stage

        backlash = 256

        camera.resolution=(640,480)
        stage.backlash = backlash

        n_moves = args.n_moves
        n_displacement = args.n_displacement

        camera.start_preview(resolution=(640,480))
        
        initial_stage_position = stage.position

        stage.move_rel([-backlash, -backlash, -backlash])
        stage.move_rel([backlash, backlash, backlash])
        
        experiment_group = df.new_group("repeatability", "Repeatability measurements for different distances in each group")

        for dist in [2**i for i in np.linspace(4, 14, n_displacement)]:
            data_gr = df.new_group("distance", "Repeatability measurements, moving the stage away and back again", parent=experiment_group)
            data_gr.attrs['move_distance'] = dist

            image = ms.rgb_image().astype(np.float32)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mean = np.mean(image)
            templ8 = (image - mean)[144:-144, 144:-144]
            data_gr['template_image'] = templ8
            data_gr['sample_image'] = image
            imgfile_location = "/home/pi/summer/drift/calibration/repeat_templ8_%s.jpg" % dist
            cv2.imwrite(imgfile_location, templ8)
            imgfile_location = "/home/pi/summer/drift/calibration/repeat_image_%s.jpg" % dist
            cv2.imwrite(imgfile_location, image)
            img = Image.open("/home/pi/summer/drift/calibration/repeat_templ8_%s.jpg" % dist)
            pad = Image.new('RGB', (352, 192)) #Tuple must be multiples of 32 and 16
            pad.paste(img, (0, 0))
            overlay = camera.add_overlay(pad.tobytes(), size = (352, 192))
            overlay.alpha = 128
            overlay.fullscreen = False
            overlay.layer = 3

            start_t = time.time()
            
            templ8_position = measure_txy(start_t, ms, templ8)
            
            def move_overlay(cx, cy):
                """move the overlay to show a shift of cx,cy camera pixels"""
                x = int(960 + (cx - templ8_position[0, 1] - 176)*2.25)
                y = int(540 + (cy - templ8_position[0, 2] - 96)*2.25)
                overlay.window = (x, y, int(352*2.25), int(192*2.25))

            for j in range(n_moves):
                move_group = df.new_group("move", "One move away and back again", parent = data_gr)
                move_group['init_stage_position'] = stage.position
                process = measure_txy(start_t, ms, templ8)
                move_group['init_cam_position'] = process
                move_overlay(*process[0, 1:3])
                move_vect = random_point(dist)
                stage.move_rel(move_vect)
                time.sleep(1)
                move_group['moved_stage_position'] = stage.position
                process = measure_txy(start_t, ms, templ8)
                move_group['moved_cam_position'] = process
                move_overlay(*process[0, 1:3])
                stage.move_rel(np.negative(move_vect))
                time.sleep(1)
                move_group['final_stage_position'] = stage.position
                process = measure_txy(start_t, ms, templ8)
                move_group['final_cam_position'] = process
                move_overlay(*process[0, 1:3])
                stage.move_abs(initial_stage_position)
            camera.remove_overlay(overlay)

        camera.stop_preview()
        print("Done")
