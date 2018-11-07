
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
from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Characterises orthogonality of the axis and finds step size")
    parser.add_argument("side_length", type=int, help="Total displacement along each axis")
    parser.add_argument("--points", type=int, default=10, help="Number of measurments to record")
    args = parser.parse_args()

    with load_microscope("microscope_settings.npz", dummy_stage = False) as ms, \
         closing(data_file.Datafile(filename = "step_size.hdf5")) as df:
        
        assert picamera_supports_lens_shading(), "You need the updated picamera module with lens shading!"

        camera = ms.camera
        stage = ms.stage

        side_length = args.side_length
        points = args.points
        backlash = 256

        camera.resolution=(640,480)
        stage.backlash = backlash

        stage_pos = df.new_group("data_stage", "orthogonality")
        cam_pos = df.new_group("data_cam", "orthogonality")
        steps = df.new_group("data_steps", "step_size")
        distance = df.new_group("data_distance", "step_size")
        data_stage = np.zeros((2 * points, 3))
        data_cam = np.zeros((2 * points, 2))
        data_steps = np.zeros(2 * (points - 1))
        data_distance = np.zeros(2 * (points - 1))

        camera.start_preview(resolution=(640,480))

        stage.move_rel([-backlash, -backlash, -backlash])
        stage.move_rel([backlash, backlash, backlash])

        image = ms.rgb_image().astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean = np.mean(image)
        templ8 = (image - mean)[144:-144, 144:-144]
        imgfile_location = "/home/pi/summer/drift/calibration/step_size_templ8.jpg"
        cv2.imwrite(imgfile_location, templ8)
        imgfile_location = "/home/pi/summer/drift/calibration/step_size_image.jpg"
        cv2.imwrite(imgfile_location, image)
        img = Image.open("/home/pi/summer/drift/calibration/step_size_templ8.jpg")
        pad = Image.new('RGB', (352, 192)) #Tuple must be multiples of 32 and 16
        pad.paste(img, (0, 0))
        overlay = camera.add_overlay(pad.tobytes(), size = (352, 192))
        overlay.alpha = 128
        overlay.fullscreen = False
        overlay.layer = 3

        initial_stage_position = stage.position
        frame = ms.rgb_image().astype(np.float32)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        templ8_position = np.zeros((1, 2))
        templ8_position[0, :], corr = find_template(templ8, frame - np.mean(frame), return_corr = True, fraction=0.5)
        
        def move_overlay(cx, cy):
            """move the overlay to show a shift of cx,cy camera pixels"""
            x = int(960 + (cx - templ8_position[0, 0] - 176)*2.25)
            y = int(540 + (cy - templ8_position[0, 1] - 96)*2.25)
            overlay.window = (x, y, int(352*2.25), int(192*2.25))

        stage.move_rel([-side_length/2, 0, -side_length/2])

        for i in range(points):
            stage.move_rel([side_length / (points - 1), 0, 0])
            data_stage[i, :] = stage.position
            frame = ms.rgb_image().astype(np.float32)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            data_cam[i, :], corr = find_template(templ8, frame - np.mean(frame), return_corr = True, fraction=0.5)
            move_overlay(data_cam[i, 0], data_cam[i, 1])
            camera.stop_preview()
            cv2.imshow("corr", corr * 255.0 / np.max(corr))
            cv2.waitKey(1000)
            camera.start_preview()
            if i > 0:
                data_steps[i - 1] = i * side_length / (points - 1)
                data_distance[i - 1] = np.sqrt((data_cam[i, 0] - data_cam[0, 0])**2 + (data_cam[i, 1] - data_cam[0, 1])**2)
        for j in range(points):
            stage.move_rel([0, 0, side_length / (points - 1)])
            i = j + points
            data_stage[i, :] = stage.position
            frame = ms.rgb_image().astype(np.float32)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            data_cam[i, :], corr = find_template(templ8, frame - np.mean(frame), return_corr = True, fraction=0.5)
            move_overlay(data_cam[i, 0], data_cam[i, 1])
            camera.stop_preview()
            cv2.imshow("corr", corr * 255.0 / np.max(corr))
            cv2.waitKey(1000)
            camera.start_preview()
            if j > 0:
                data_steps[i - 2] = j * side_length / (points - 1)
                data_distance[i - 2] = np.sqrt((data_cam[i, 0] - data_cam[points, 0])**2 + (data_cam[i, 1] - data_cam[points, 1])**2)

        df.add_data(data_stage, stage_pos, "data_stage")
        df.add_data(data_cam, cam_pos, "data_cam")
        df.add_data(data_steps, steps, "data_steps")
        df.add_data(data_distance, distance, "data_distance")

        stage.move_abs(initial_stage_position)
        camera.stop_preview()
        print("Done")
