"""
Usage (at the top directory): python -m demo.create_random_mismatch_traj
"""

import os
import argparse
import numpy as np
import json
import copy
from PIL import Image

from .constants import get_demo_dir, get_demo_gif_path

"""
File structure to store the randomly generated mismatch

Trim the demo after being successful consecutively for 10% of total timesteps

- task_name
    - random_mismatched
        - 1outof5_mismatched
            - 1outof5_mismatched_0
            - 1outof5_mismatched_1
            - 1outof5_mismatched_2
        - 2outof5_mismatched
            - ...
        - 3outof5_mismatched
            - ...
"""

MISMATCH_SPEED_LEVELS = {
    "fast": [2, 4, 6, 8, 10],
    "slow": [-1, -2, -3, -4, -5],  # Number of duplicates
    "mixed": []
}

def load_frames_and_states(input_gif_path, input_states_path=None):

    if input_states_path is None: # infer from gif path
        input_states_path =  os.path.splitext(input_gif_path)[0] + "_states.npy"

    assert input_gif_path.endswith(".gif"), "error: reference seq not a gif"
    
    # Load GIF and states
    gif = Image.open(input_gif_path)
    states = np.load(input_states_path)

    # Verify that the number of frames matches the states
    frames = []
    try:
        while True:
            frames.append(gif.copy())
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass  # End of GIF frames

    return frames, states

def save_frames_and_states(frames, gif_path, states, states_path, use_pil=True):
    if use_pil:
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            loop=0
        )
    else:
        import imageio

        imageio.mimsave(gif_path, frames, duration=0.1, loop=0, plugin="pillow", optimize=False, disposal=2)  # duration is in seconds

    np.save(states_path, states)


def random_subsample_gif_and_states(input_gif_path, output_dir, speed_type, mismatch_level, num_of_sections, pct_of_success_needed=0.1):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_gif_path))[0]
    frames, states = load_frames_and_states(input_gif_path)

    # Load the success vector
    success = np.load(os.path.splitext(input_gif_path)[0] + "_success.npy")
    # Find the beginning of the first N consecutive successes
    N_consecutive_success_needed = int(len(success) * pct_of_success_needed)
    start_idx = 0
    found_N_consecutive = False
    while start_idx < len(success) - N_consecutive_success_needed:
        if np.all(success[start_idx:start_idx + N_consecutive_success_needed]):
            found_N_consecutive = True
            break
        start_idx += 1

    if not found_N_consecutive:
        print(f"Could not find {N_consecutive_success_needed} consecutive successes in {input_gif_path}")
    
    frames = frames[:start_idx + N_consecutive_success_needed]
    states = states[:start_idx + N_consecutive_success_needed]

    print(f"Trimming the demo to {len(frames)} frames after {N_consecutive_success_needed} consecutive successes")

    num_frames = len(frames)
    num_frames_per_section = num_frames // num_of_sections

    # Randomly select the section to mismatch
    mismatched_section = np.random.choice(num_of_sections, mismatch_level, replace=False)
    mismatched_section.sort()

    mismatched_selected_idx = []
    mismatched_info = {}

    # Mismatch the selected sections
    for i in range(num_of_sections):
        if i in mismatched_section:
            # Randomly select a subsample speed level
            if mismatch_level == 5:
                if i == 0:
                    # Without replacement
                    available_speed_levels = copy.deepcopy(MISMATCH_SPEED_LEVELS[speed_type])

                speed_level = np.random.choice(available_speed_levels)   
                # Pop the selected speed level
                available_speed_levels.remove(speed_level) 
            else:
                available_speed_levels = [level for level in MISMATCH_SPEED_LEVELS[speed_type] if level < num_frames_per_section]

                speed_level = np.random.choice(available_speed_levels)

            if speed_level < 0:
                # We duplicate the frames consecutively
                # Original indices is 1, 2, 3, 4, 5
                # And num_of_dup=1, then the duplicated indices is 1, 1, 2, 2, 3, 3, 4, 4, 5, 5
                num_of_dup = abs(speed_level)

                subsampled_indices = list(range(i * num_frames_per_section, (i + 1) * num_frames_per_section))
                subsampled_indices = [idx for idx in subsampled_indices for _ in range(num_of_dup + 1)]
            else:
                subsampled_indices = list(range(i * num_frames_per_section, (i + 1) * num_frames_per_section))[::speed_level]

            print(f"i={i}, speed_level={speed_level}, len={len(subsampled_indices)}, {subsampled_indices}")

            mismatched_selected_idx.extend(subsampled_indices)

            mismatched_info[i] = {
                'speed_level': int(speed_level),
                'subsampled_indices': subsampled_indices
            }
        else:
            mismatched_selected_idx.extend(list(range(i * num_frames_per_section, (i + 1) * num_frames_per_section)))

            mismatched_info[i] = {
                'speed_level': -1,
                'subsampled_indices': list(range(i * num_frames_per_section, (i + 1) * num_frames_per_section))
            }

        print(f"i={i}, current len={len(mismatched_selected_idx)}")

    if len(frames) - 1 not in mismatched_selected_idx:
        mismatched_selected_idx.append(len(frames) - 1)

    print(f"[len={len(mismatched_selected_idx)}] Saving the mismatched trajectory at {output_dir}")
    print("=====================")

    mismatched_frames = []
    prev_idx = -1
    for idx in mismatched_selected_idx:
        frame_arr = np.array(frames[idx].convert('RGB'))
        if idx == prev_idx:
            # Add minor perturbation because python gif library (PIL/Imageio) does not support duplicate frames
            episilon = 0.00000001
            # noise = np.random.uniform(-episilon, episilon, frame_arr.shape).astype(np.float32)
            noise = np.zeros(frame_arr.shape)
            # Randomly pick a channel to perturb
            noise[:, :, np.random.randint(0, frame_arr.shape[2])] = np.random.uniform(-episilon, episilon, frame_arr.shape[:2]).astype(np.float32)

            # TODO: perturbing one frame is not enough. Adding all the noise might produce too much difference in resnet encoding though. Need to try
            # noise = np.zeros(frame_arr.shape)
            # perturb_idx = (np.random.randint(0, frame_arr.shape[0]), np.random.randint(0, frame_arr.shape[1]), np.random.randint(0, frame_arr.shape[2]))
            # noise[perturb_idx] = np.random.uniform(-episilon, episilon)

            frame_arr = np.clip(frame_arr + noise * 255, 0, 255).astype(np.uint8)
        
        mismatched_frames.append(frame_arr)
        prev_idx = idx

    mismatched_frames = [Image.fromarray(arr) for arr in mismatched_frames]
    # mismatched_frames = [np.array(frames[i].convert("RGB")) for i in mismatched_selected_idx]
    mismatched_states = np.array([states[i] for i in mismatched_selected_idx])

    print(f"len(mismatched_frames)={len(mismatched_frames)}, len(mismatched_states)={len(mismatched_states)}")

    # input("before saving")

    # Save the mismatched trajectory
    save_frames_and_states(mismatched_frames, os.path.join(output_dir, f"{base_name}.gif"), mismatched_states, os.path.join(output_dir, f"{base_name}_states.npy"), use_pil = 'fast' == speed_type)

    saved_frames, saved_states = load_frames_and_states(os.path.join(output_dir, f"{base_name}.gif"), os.path.join(output_dir, f"{base_name}_states.npy"))

    if len(saved_frames) != len(mismatched_frames) or len(saved_states) != len(mismatched_states) or len(saved_frames) != len(saved_states):
        raise ValueError(f"Error in saving the mismatched trajectory: len(saved_frames)={len(saved_frames)}, len(saved_states)={len(saved_states)}")

    # Save the mismatched info as json
    with open(os.path.join(output_dir, f"{base_name}_mismatched_info.json"), 'w') as f:
        json.dump(mismatched_info, f, indent=4)

def create_random_mismatch_trajs(input_gif_path, speed_type, mismatch_level, num_of_sections, num_traj):
    for i in range(num_traj):
        output_dir = get_demo_dir(env_name, task_name, camera_name, mismatched=False, random_mismatched_info=
                                  {
                                    'speed_type': speed_type,
                                    'mismatch_level': f'{mismatch_level}outof{num_of_sections}_mismatched', 
                                    'run_num': i}) 
        
        if os.path.exists(input_gif_path):
            random_subsample_gif_and_states(input_gif_path, output_dir, speed_type, mismatch_level, num_of_sections)
        else:
            print(f"{input_gif_path} does not exist")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env_name", type=str, default="door-close-v2")
    parser.add_argument("-s", "--speed", type=str, default="fast", help="speed of trajectory")
    args = parser.parse_args()

    task_name = args.env_name # weird thing, we overloaded "env_name"
    speed_type = args.speed  # options: 'fast', 'slow', 'mixed'
    camera_name = "d"
    env_name = "metaworld"
    mismatch_level = [1, 3, 5]
    num_of_sections = 5 # How many subsection to partition the demo into
    num_traj = 3
    pct_of_frames_to_trim = 0.1

    input_gif_path = get_demo_gif_path(env_name, task_name, camera_name, demo_num=0, num_frames="d") 
    # new gif path

    if type(mismatch_level) == list:
        for level in mismatch_level:
            create_random_mismatch_trajs(input_gif_path, speed_type, level, num_of_sections, num_traj)
    elif type(mismatch_level) == int:
        create_random_mismatch_trajs(input_gif_path, speed_type, mismatch_level, num_of_sections, num_traj)