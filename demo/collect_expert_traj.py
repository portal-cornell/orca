"""
Adapted from https://github.com/fuyw/TemporalOT/blob/main/collect_expert_traj.py

Usage:
- If you just want to use the default camera (assuming that it's in the dictionary), you can run:
    python demo/collect_expert_traj.py -e hammer-v2 -c d
- If you want to use a specific camera, you can run:
    python demo/collect_expert_traj.py -e hammer-v2 -c corner3

    The options are: corner, corner2, corner3, corner4

It is helpful to double check the gif generated and the rewards/successes printed in the terminal to make sure that we are getting good demos!
"""

# An example of generating expert demos by MetaWorld's scripted policies.
# Warning: The scripted policies may generate failure trajectories. You can write a task-dependent criterion to filter the generated trajectories.

import argparse
import metaworld
import random
import metaworld.policies as policies
import cv2
import numpy as np

import pickle
import imageio
from pathlib import Path
import os
from collections import deque
from PIL import Image
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from utils.cluster_utils import set_os_vars
from .constants import CAMERA, MAX_PATH_LENGTH, get_demo_dir


set_os_vars()  # To work for G2

POLICY = {
    'button-press-v2': policies.SawyerButtonPressV2Policy, # Added by us
    'door-close-v2': policies.SawyerDoorCloseV2Policy,  # Added by us
    'hammer-v2': policies.SawyerHammerV2Policy,
    'drawer-close-v2': policies.SawyerDrawerCloseV2Policy,
    'drawer-open-v2': policies.SawyerDrawerOpenV2Policy,
    'door-open-v2': policies.SawyerDoorOpenV2Policy,
    'bin-picking-v2': policies.SawyerBinPickingV2Policy,
    'button-press-topdown-v2': policies.SawyerButtonPressTopdownV2Policy,
    'door-unlock-v2': policies.SawyerDoorUnlockV2Policy,
    'basketball-v2': policies.SawyerBasketballV2Policy,
    'basketball-v3': policies.SawyerBasketballV2Policy,
    'plate-slide-v2': policies.SawyerPlateSlideV2Policy,
    "hand-insert-v2": policies.SawyerHandInsertV2Policy,  
    "peg-insert-side-v2": policies.SawyerPegInsertionSideV2Policy,  
    'push-wall-v2': policies.SawyerPushWallV2Policy,
    'soccer-v2': policies.SawyerSoccerV2Policy,
    'assembly-v2': policies.SawyerAssemblyV2Policy,
    'disassemble-v2': policies.SawyerDisassembleV2Policy,
    'pick-place-wall-v3': policies.SawyerPickPlaceWallV2Policy,
    'pick-place-v2': policies.SawyerPickPlaceV2Policy,
    'push-v2': policies.SawyerPushV2Policy,
    'lever-pull-v2': policies.SawyerLeverPullV2Policy,
    'stick-pull-v2': policies.SawyerStickPullV2Policy,
    'shelf-place-v2': policies.SawyerShelfPlaceV2Policy,
    'window-close-v2': policies.SawyerWindowCloseV2Policy,
    'window-open-v2': policies.SawyerWindowOpenV2Policy,
    'reach-v2': policies.SawyerReachV2Policy,
    'button-press-wall-v2': policies.SawyerButtonPressWallV2Policy,
    'box-close-v2': policies.SawyerBoxCloseV2Policy,
    'stick-push-v2': policies.SawyerStickPushV2Policy,
    'handle-pull-v2': policies.SawyerHandlePullV2Policy,
    'door-lock-v2': policies.SawyerDoorLockV2Policy,
}


# For some tasks, we want to stop the robot after the goal is achieved
CLEAR_ACTION_AFTER_SUCCESS = {
    'lever-pull-v2': True,  # If not, the hand-engineered policy will keep pushing, causing the lever to go beyond the goal
}


def record_video(fname, large_images):
    # Save the video as a gif
    imageio.mimsave(f"{fname}.gif", large_images, duration=1/30, loop=0)

def collect_trajectories(env_name, num_demos, camera_name, skip_frames=0):
    # For some tasks, we want to stop the robot after the goal is achieved
    #   (We specifically set the action to zero)
    stop = CLEAR_ACTION_AFTER_SUCCESS.get(env_name, False)

    env_folder = get_demo_dir("metaworld", env_name, camera_name)
    # Make the folder if it doesn't exist
    Path(env_folder).mkdir(parents=True, exist_ok=True)

    policy = POLICY[env_name]()
    env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[f"{env_name}-goal-observable"](render_mode="rgb_array", 
                                                                            camera_name=camera_name,
                                                                            episode_length=MAX_PATH_LENGTH[env_name],
                                                                            env_reward_type="dense",
                                                                            render_dim=(480, 480))
    env._freeze_rand_vec = False


    episode = 0
    while episode < num_demos:
        print(f"==================================== Episode {episode}, {env_name}, camera={camera_name} ====================================")
        images = list()
        large_images = list()
        observations = list()
        actions = list()
        rewards = list()
        successes = list()
        image_stack = deque([], maxlen=3)
        large_imae_stack = deque([], maxlen=3)
        goal_achieved = 0

        # Reset env
        observation, info = env.reset()  # Reset environment
        
        num_steps = MAX_PATH_LENGTH[env_name]
        for step in range(num_steps):
            # Get action
            if stop and goal_achieved:
                # If we want to do nothing after the goal is achieved
                action = np.zeros_like(actions[-1])
            else:
                action = policy.get_action(observation)
                action = np.clip(action, -1.0, 1.0)
            actions.append(action)

            # Get observation
            observation[-3:] = 0
            observations.append(observation)

            # Act in the environment
            observation, reward, done, truncated, info = env.step(action)
            rewards.append(reward)
            successes.append(int(info['success']))
            goal_achieved = max(int(info['success']), goal_achieved)

            # Get frames
            screen = env.render()

            if camera_name == "corner" or camera_name == "corner2" or camera_name == "corner3":
                # For some reason, the image is flipped upside down
                screen = np.flipud(screen)
            elif camera_name == "corner4":
            # For some reason, the image is flipped left-right
                screen = np.fliplr(screen)

            image_int = np.uint8(screen)[:env._render_dim[0], :env._render_dim[1], :]
            # if camera_name == "corner" or camera_name == "corner2" or camera_name == "corner3":
            #     # These 3 cameras are upside down
            #     image_int = np.flipud(image_int)
            # elif camera_name == "corner4":
            #     # This camera is left-right flipped
            #     image_int = np.fliplr(image_int)
                
            large_images.append(Image.fromarray(image_int))

            print(f"Ep {episode}, Step {step}, reward: {reward:.2f}, goal_achieved/success: {goal_achieved}")

        ep_name = f"{env_name}_{camera_name}_{episode}"

        record_video(f"{env_folder}/{ep_name}", large_images[::skip_frames+1])

        # You can write a task-dependent criterion to filter the generated trajectories!!!
        if goal_achieved == 0:
            print(f"Failed episode {episode} for {env_name}")
            input("stop")

        # Store trajectory
        episode = episode + 1

        # Adding [::2] as striding for action repeat agent

        # Save the observations
        with open(f"{env_folder}/{ep_name}_states.npy", "wb") as f:
            # Originally it's (num_steps, obs_size), but seq_utils.py expects (num_steps, n_envs, obs_size)
            observations = np.array(observations)[:, np.newaxis, :][::skip_frames+1]
            print(f"Saving states with shape {observations.shape}")
            np.save(f, observations)

        # Save the actions
        with open(f"{env_folder}/{ep_name}_actions.npy", "wb") as f:
            print(f"Saving actions with shape {np.array(actions).shape}")
            np.save(f, np.array(actions)[::2])

        # Save the rewards
        with open(f"{env_folder}/{ep_name}_rewards.npy", "wb") as f:
            print(f"Saving rewards with shape {np.array(rewards).shape}")
            np.save(f, np.array(rewards)[::2])

        # Save the sparse reward (success)
        with open(f"{env_folder}/{ep_name}_success.npy", "wb") as f:
            print(f"Saving success with shape {np.array(successes).shape}")
            np.save(f, np.array(successes)[::2])
        print(f"==================================== Episode {episode}, {env_name}, camera={camera_name} ====================================")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env_name", type=str, default="door-close-v2")
    parser.add_argument("-n", "--num_demos", type=int, default=1)
    parser.add_argument("-c", "--camera_name", type=str, default="d", choices=["d", "corner", "corner2", "corner3", "corner4"])
    parser.add_argument("-s", "--skip_frames", type=int, default=1, help="Number of frames to skip (0 means not skipping any frame; 1 means getting every other frame)")
    args = parser.parse_args()

    num_demos = args.num_demos
    env_name = args.env_name

    # "d" is a placeholder for the default camera
    if args.camera_name != "d":
        camera_name = args.camera_name
    else:
        camera_name = CAMERA[env_name]


    collect_trajectories(env_name, num_demos, camera_name, skip_frames=args.skip_frames)
