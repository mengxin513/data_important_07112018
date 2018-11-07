
from __future__ import print_function
from builtins import input
import io
import sys
import os
import time
import argparse
import numpy as np
from openflexure_stage import OpenFlexureStage
from openflexure_microscope import load_microscope
from openflexure_microscope.microscope import picamera_supports_lens_shading
import matplotlib.pyplot as plt
from contextlib import closing
import data_file
from camera_stuff import find_template
import threading
import queue
import cv2

def move_stage(step, dwell_time, event, stage, moves=[]):
    while not event.wait(dwell_time):
        moves.append([time.time(),] + list(stage.position))
        stage.move_rel(step)
        moves.append([time.time(),] + list(stage.position))

def printProgressBar(iteration, total, length = 10):
    percent = 100.0 * iteration / total
    filledLength = int(length * iteration // total)
    bar = '*' * filledLength + '-' * (length - filledLength)
    print('Progress: |%s| %d%% Completed' % (bar, percent), end = '\r')
    if iteration == total: 
        print()
        
def move_stage_and_record(step, N_frames, microscope, data_group, template, dwell_time):
    # move the stage by a given step, recording motion as we go.
    ms = microscope
    data_group.attrs['step'] = step
    ms.camera.start_preview(resolution=(640,480))
    
    # we will use a RAM buffer to record a bunch of frames
    outputs = [io.BytesIO() for i in range(N_frames)]
    stop_moving_event = threading.Event()
    stage_moves = []
    movement_thread = threading.Thread(target = move_stage, 
                                       args = (step, dwell_time, stop_moving_event, ms.stage, stage_moves), 
                                       name = 'stage_movement_thread')
    movement_thread.start()
    print("Starting acquisition of {} frames, should take about {}s.".format(N_frames, N_frames / ms.camera.framerate))
    try:
        start_t = time.time()
        camera.capture_sequence(outputs, 'jpeg', use_video_port=True)
        end_t = time.time()
    finally:
        print ("Stopping...")
        stop_moving_event.set()
        movement_thread.join()
        ms.camera.stop_preview()

    print("Recorded {} frames in {} seconds ({} fps)".format(N_frames, end_t - start_t, N_frames / (end_t - start_t)))
    print("Camera framerate was set to {}, and reports as {}".format(framerate, camera.framerate))
    
    data_group['stage_moves'] = np.array(stage_moves)
    data_group['stage_moves'].attrs['description'] = "t,x,y,z data for the stage's motion during the sequence of moves " \
                                                     "time is in seconds, position is in stage steps."
    # go through the captured images and process them.
    data = np.zeros((N_frames, 3))
    for j, k in enumerate(outputs):
            frame_data = np.fromstring(k.getvalue(), dtype = np.uint8)
            frame = cv2.imdecode(frame_data, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            data[j, 1:], corr = find_template(template, frame - np.mean(frame), return_corr = True, fraction = 0.5)
            data[j, 0] = float(j) / float(framerate) + start_t
            printProgressBar(j, N_frames)
    print("")

    data_group["camera_motion"] = data
    data_group["camera_motion"].attrs['description'] = "t,x,y data of the position, as recorded by the camera." \
                                                       "t is in seconds, x/y are in pixels"
    data_group["camera_motion"].attrs['reported_framerate'] = ms.camera.framerate
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform a linear motion test on an Openflexure Microscope")
    parser.add_argument("step", type=int, nargs=3, help="The displacement between each point, in steps")
    parser.add_argument("--n_frames", type=int, default=2000, help="The number of frames to record for each run")
    parser.add_argument("--n_repeats", type=int, default=1, help="How many there-and-back round trips to do")
    parser.add_argument("--framerate", type=int, default=90, help="Rate at which to run the camera (frames/second)")
    parser.add_argument("--dwell_time", type=float, default=1.0, help="Time (in seconds) to wait at each point")
    parser.add_argument("--return_to_start", dest="return_to_start", action="store_true", help="Return to the origin at the beginning of each run")
    parser.add_argument("--output", help="HDF5 file to save to", default="linear_motion.hdf5")
    parser.add_argument("--settings_file", help="File where the microscope settings are stored.", default="microscope_settings.npz")
    args = parser.parse_args()

    with load_microscope(args.settings_file, dummy_stage = False) as ms, \
         closing(data_file.Datafile(filename = args.output)) as df:

        assert picamera_supports_lens_shading(), "You need the updated picamera module with lens shading!"

        camera = ms.camera
        stage = ms.stage

        N_frames = args.n_frames
        N_repeats = args.n_repeats
        step = args.step
        framerate = args.framerate
        dwell_time = args.dwell_time
        backlash = 256
        return_to_start = args.return_to_start

        camera.resolution=(640,480)
        camera.zoom=(0,0,1,1)
        camera.framerate = framerate
        stage.backlash = backlash

        camera.start_preview(resolution=(640,480))
        initial_stage_position = stage.position

        image = ms.rgb_image().astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean = np.mean(image)
        template = (image - mean)[100:-100, 100:-100]
        
        output_group = df.new_group("stepwise_motion", "Moving the stage along a line, recording position as we go.")
        output_group['template_image'] = template
        output_group['sample_image'] = ms.rgb_image()
        output_group.attrs['step'] = step
        output_group.attrs['n_repeats'] = N_repeats
        output_group.attrs['n_frames'] = N_frames
        output_group.attrs['requested_framerate'] = framerate
        output_group.attrs['backlash'] = backlash
        output_group.attrs['return_to_start'] = return_to_start
        
        # (fairly basic) backlash correction: we'll approach the starting point from the right direction
        stage.move_rel([-backlash, -backlash, -backlash])
        for i in range(N_repeats): # move back and forth 5 times
            if i==0 or return_to_start:
                # NB backlash correction should kick in automatically when this move happens if needed
                stage.move_abs(initial_stage_position)
            g = output_group.create_group("sequence_{:05}".format(i))
            move_stage_and_record(step, N_frames, ms, g, template, dwell_time)
            
        stage.move_abs(initial_stage_position)
        camera.stop_preview()
