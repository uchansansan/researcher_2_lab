#!/usr/bin/env python
# manual

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""
# from PIL import Image
import argparse
import sys

import gym
import numpy as np
import cv2 as cv 
import pyglet
from pyglet.window import key

from gym_duckietown.envs import DuckietownEnv

# from experiments.utils import save_img

import cv2 as cv
import numpy as np
from cv2 import aruco
core = np.ones([3,3])

class VideoDetection:
    def __init__(self, visualization=False):
        self.red_low = np.array([150, 150, 255])
        self.red_high = np.array([13, 16, 153])
        self.vizualization = visualization 

    def scan_lines(self, img):
        mask = self.split_mask(self.sdelat_krasivo(cv.inRange(img, self.red_high, self.red_low)))
        # msk = mask + np.zeros((self.y, self.x))[self.diff_y:, self.diff_x:-self.diff_x]
        
        edges = cv.Canny(mask, 50, 150)
        lines = cv.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=50, maxLineGap=150)
        return lines


    def scan_code(self, frame):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_APRILTAG_36h11)
        parameters =  aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        return corners, ids

    def split_mask(self, mask):
        y,x = mask.shape
        diff_y, diff_x = round(y/1.4), round(x/7) 
        self.y, self.x, self.diff_x, self.diff_y = y,x, diff_x, diff_y

        cropped_mask = mask.copy()[ diff_y:, diff_x:-diff_x]
        # cv.imshow("mask", cropped_mask)
        
        return cropped_mask

    def sdelat_krasivo(self, mask):
        er = cv.erode(mask, core, iterations=2)
        dil = cv.dilate(er, core, iterations=4)
        return dil

    def scan(self, img):
        frame_markers = img
        ids = None

        lines = self.scan_lines(img)
        if(lines is not None):
            if(len(lines) >= 2):
                corners, ids = self.scan_code(img)
        if (self.vizualization):
            if(lines is not None):
                if(len(lines) >=2):
                    if(ids is not None):
                        frame_markers = aruco.drawDetectedMarkers(img.copy(), corners, ids)
                    for line in lines:
                        x1,y1,x2,y2 = line[0]
                        cv.line(frame_markers,(x1+self.diff_x,y1+self.diff_y),(x2+self.diff_x,y2+self.diff_y),(0,255,0),2)
        cv.rectangle(frame_markers, (self.x-self.diff_x, self.y ), (self.diff_x,self.diff_y),(255,0,0), thickness=2)
        return ids, frame_markers

parser = argparse.ArgumentParser()
parser.add_argument("--env-name", default=None)
parser.add_argument("--map-name", default="udem1")
parser.add_argument("--distortion", default=False, action="store_true")
parser.add_argument("--camera_rand", default=False, action="store_true")
parser.add_argument("--draw-curve", action="store_true", help="draw the lane following curve")
parser.add_argument("--draw-bbox", action="store_true", help="draw collision detection bounding boxes")
parser.add_argument("--domain-rand", action="store_true", help="enable domain randomization")
parser.add_argument("--dynamics_rand", action="store_true", help="enable dynamics randomization")
parser.add_argument("--frame-skip", default=1, type=int, help="number of frames to skip")
parser.add_argument("--seed", default=1, type=int, help="seed")
args = parser.parse_args()

if args.env_name and args.env_name.find("Duckietown") != -1:
    env = DuckietownEnv(
        seed=args.seed,
        map_name=args.map_name,
        draw_curve=args.draw_curve,
        draw_bbox=args.draw_bbox,
        domain_rand=args.domain_rand,
        frame_skip=args.frame_skip,
        distortion=args.distortion,
        camera_rand=args.camera_rand,
        dynamics_rand=args.dynamics_rand,
    )
else:
    env = gym.make(args.env_name)

env.reset()
env.render()


@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print("RESET")
        env.reset()
        env.render()
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)

detect = VideoDetection(visualization=True)

# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)



def update(dt):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """
    wheel_distance = 0.102
    min_rad = 0.08

    action = np.array([0.0, 0.0])

    if key_handler[key.UP]:
        action += np.array([0.44, 0.0])
    if key_handler[key.DOWN]:
        action -= np.array([0.44, 0])
    if key_handler[key.LEFT]:
        action += np.array([0, 1])
    if key_handler[key.RIGHT]:
        action -= np.array([0, 1])
    if key_handler[key.SPACE]:
        action = np.array([0, 0])

    v1 = action[0]
    v2 = action[1]
    # Limit radius of curvature
    if v1 == 0 or abs(v2 / v1) > (min_rad + wheel_distance / 2.0) / (min_rad - wheel_distance / 2.0):
        # adjust velocities evenly such that condition is fulfilled
        delta_v = (v2 - v1) / 2 - wheel_distance / (4 * min_rad) * (v1 + v2)
        v1 += delta_v
        v2 -= delta_v

    action[0] = v1
    action[1] = v2

    # Speed boost
    if key_handler[key.LSHIFT]:
        action *= 1.5

    obs, reward, done, info = env.step(action)
    obs = cv.cvtColor(obs, cv.COLOR_BGR2RGB)
    ids, frame =  detect.scan(obs)
    cv.imshow("game", frame)



    print("step_count = %s, reward=%.3f" % (env.unwrapped.step_count, reward))

    # if key_handler[key.RETURN]:

    #     im = Image.fromarray(obs)

    #     im.save("screen.png")

    if done:
        print("done!")
        env.reset()
        env.render()

    env.render()


pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()
