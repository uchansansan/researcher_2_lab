import argparse
import sys
import gym
import pyglet
from pyglet.window import key
from gym_duckietown.envs import DuckietownEnv
import libs.map_builder as mpbr
import libs.VideoDetection as viddetect
import cv2 as cv
import numpy as np
from cv2 import aruco
import time

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

# INITIALIZING OUR CLASSES #
detect = viddetect.VideoDetection(visualization=True)
duck = None
# INITIALIZING OUR CLASSES #

# INITIALIZING CONSTANTS #
MANUAL = False
# INITIALIZING CONSTANTS #

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



def update(dt):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """
    global MANUAL, duck, detect

    if duck is None:
        duck = mpbr.DuckAI([env.cur_pos, env.cur_angle])

    wheel_distance = 0.102
    min_rad = 0.08

    action = np.array([0.0, 0.0])
    if not MANUAL:
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