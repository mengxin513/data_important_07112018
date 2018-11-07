
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

if __name__ == "__main__":

    with load_microscope("microscope_settings.npz") as ms:

        assert picamera_supports_lens_shading(), "You need the updated picamera module with lens shading!"

        camera = ms.camera
        stage = ms.stage

        camera.resolution=(640,480)

        camera.start_preview(resolution=(640,480))
        
        i = 0.1

        while i < 1:
            image = ms.rgb_image().astype(np.float32)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mean = np.mean(image)
            templ8 = (image - mean)[100:-100, 100:-100]
        
            frame = image
            pos, corr = find_template(templ8, frame - np.mean(frame), return_corr = True, fraction = i)
        
            plt.figure()
            plt.plot(np.sum(corr, axis=1))
            plt.savefig("test_correlation_1_fraction_{}.jpg".format(i), bbox_inches='tight', dpi=180)
            plt.figure()
            plt.plot(np.sum(corr, axis=0))
            plt.savefig("test_correlation_2_fraction_{}.jpg".format(i), bbox_inches='tight', dpi=180)
            plt.figure()
            plt.imshow(corr.astype(np.float))
            plt.savefig("test_correlation_3_fraction_{}.jpg".format(i), bbox_inches='tight', dpi=180)
            plt.close('all')
            i += 0.1
