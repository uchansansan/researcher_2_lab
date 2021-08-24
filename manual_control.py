import argparse
import sys
import gym
import pyglet
from pyglet.window import key
from gym_duckietown.envs import DuckietownEnv
import cv2 as cv
import numpy as np
from cv2 import aruco
import time

class VideoDetection:
    def __init__(self, visualization=False):
        self._SQUARE_BOUND = 3000
        self._DELAY = 50
        self._RED_LOW = np.array([150, 150, 255])
        self._RED_HIGH = np.array([13, 16, 153])
        self._VIZUALIZATION = visualization 

        self._is_scanned = False
        self._timer = 0

        self._core = np.ones([3,3])

    def scan(self, img, tick):
        frame_markers = img
        ids = None
        lines = self._scan_lines(img)

        if self._is_scanned and tick-self._timer >= self._DELAY:
            self._is_scanned = False
            self._timer = tick
            return ids, frame_markers
            
        if lines is not None and len(lines) >= 2: 
            corners, ids = self._scan_code(img)
            square = self._GaussSquare(corners)

            if square  < self._SQUARE_BOUND:
                ids = None

            if self._is_scanned:
                ids = None
                
            if ids is not None:
                if not self._is_scanned:
                    self._timer = tick
                    self._is_scanned = True

            if self._VIZUALIZATION:
                if ids is not None:
                    frame_markers = self._draw_code_box(corners, ids, frame_markers)
                frame_markers = self._draw_lines(lines, frame_markers)

        cv.rectangle(frame_markers, (self.x-self.diff_x, self.y ), (self.diff_x,self.diff_y),(255,0,0), thickness=2)
        return ids, frame_markers

    def _draw_lines(self, lines, image):
        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv.line(image,(x1+self.diff_x,y1+self.diff_y),(x2+self.diff_x,y2+self.diff_y),(0,255,0),2)
        return image

    def _draw_code_box(self, corners, ids, image):
        frame_markers = aruco.drawDetectedMarkers(image.copy(), corners, ids)
        return frame_markers

    def _scan_lines(self, img):
        mask = self._split_mask(self._sdelat_krasivo(cv.inRange(img, self._RED_HIGH, self._RED_LOW)))
        edges = cv.Canny(mask, 50, 150)
        lines = cv.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=50, maxLineGap=150)
        return lines

    def _scan_code(self, frame):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_APRILTAG_36h11)
        parameters =  aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        return corners, ids

    def _GaussSquare(self, vertex):
        right, left, res = 0,0,0
        if vertex != []:
            for i in range(len(vertex[0][0])):
                if i+1 < len(vertex[0][0]):
                    right += vertex[0][0][i][0] * vertex[0][0][i+1][1]
                    left += vertex[0][0][i][1] * vertex[0][0][i+1][0]
            res = right - left
        return abs(res)

    def _split_mask(self, mask):
        y,x = mask.shape
        diff_y, diff_x = round(y/1.4), round(x/7) 
        self.y, self.x, self.diff_x, self.diff_y = y,x, diff_x, diff_y
        cropped_mask = mask.copy()[ diff_y:, diff_x:-diff_x]
        return cropped_mask

    def _sdelat_krasivo(self, mask):
        er = cv.erode(mask, self._core, iterations=2)
        dil = cv.dilate(er, self._core, iterations=4)
        return dil

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

def regulator(w,r):
    kp = 1
    kRot= 1
    k_lin = 1/2
    k_ang = 4.7
    V0 = 0.45

    V_L = V0 + kp * r+ kRot*w
    V_R = V0 - kp * r- kRot*w

    V_LIN = k_lin * (V_L + V_R)
    V_ANG = k_ang * (V_L - V_R)
    return np.array([V_LIN, V_ANG])

MAXTICKS = 50
current_ticks = MAXTICKS
left  = False
right = False
up    = False

def naprav(ukaz):
    global  left, right, up
    if ukaz == 0:
        left = True
        return np.array([0,1])
    elif ukaz == 1:
        right = True
        return np.array([1,0])
    elif ukaz == 2:
        up = True
        return np.array([0,-1])

def ezda(*args):
    global left, right, up, current_ticks
    if left or right or up:
        current_ticks -= 1
        if current_ticks < 1:
            left, right, up = False, False, False
            current_ticks = MAXTICKS
    if left:
        return naprav(0)
    elif right:
        return naprav(2)
    elif up:
        return naprav(1)
    elif len(args)==2:
        return regulator(args[0], args[1])
    else:
        return naprav(args[2])

Manual = False


def update(dt):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """
    global Manual
    wheel_distance = 0.102
    min_rad = 0.08

    action = np.array([0.0, 0.0])
    if not Manual:
        lane_pose = env.get_lane_pos2(env.cur_pos, env.cur_angle)
        dist  = lane_pose.dist
        angle = lane_pose.angle_rad

        action = regulator(angle, dist )
    else:
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
        # env.unwrapped.step_count
        action[0] = v1
        action[1] = v2

    # Speed boost
    if key_handler[key.LSHIFT]:
        action *= 1.5

    obs, reward, done, info = env.step(action)
    obs = cv.cvtColor(obs, cv.COLOR_BGR2RGB)

    ids, frame =  detect.scan(obs, env.unwrapped.step_count)

    cv.imshow("game", frame)

    if done:
        print("done!")
        env.reset()
        env.render()

    env.render()

pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()
env.close()