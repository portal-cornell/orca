import os
import numpy as np
from PIL import Image
from .constants import get_demo_dir, get_demo_gif_path

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

def evenly_subsample_gif_and_states(input_gif_path, output_dir, N, last_frame=None, cut_after_N_consecutive_success=None, input_states_path=None):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_gif_path))[0]
    frames, states = load_frames_and_states(input_gif_path, input_states_path)

    if last_frame is not None:
        frames = frames[:last_frame]
        states = states[:last_frame]
    elif cut_after_N_consecutive_success is not None:
        # Load the success vector
        success = np.load(os.path.splitext(input_gif_path)[0] + "_success.npy")
        print(f"Loaded success vector of length {len(success)}")
        # Find the beginning of the first N consecutive successes
        start_idx = 0
        found_N_consecutive = False
        while start_idx < len(success) - N:
            if np.all(success[start_idx:start_idx + N]):
                found_N_consecutive = True
                break
            start_idx += 1

        if not found_N_consecutive:
            raise ValueError(f"Could not find {N} consecutive successes in {input_gif_path}")
        
        frames = frames[:start_idx + N]
        states = states[:start_idx + N]

    num_frames = len(frames)
    if num_frames != states.shape[0]:
        raise ValueError(f"Mismatch between GIF frames ({num_frames}) and states ({states.shape[0]}) in {base_name}")

    # Subsample evenly
    step = num_frames // N
    selected_indices = list(range(0, num_frames, step))[:N]
    subsampled_frames = [frames[i] for i in selected_indices]
    subsampled_states = states[selected_indices]

    # Save subsampled frames as a new GIF
    file_base_name = f"{base_name}_subsampled_{N}"
    if cut_after_N_consecutive_success:
        file_base_name += f"_cut-after-{cut_after_N_consecutive_success}-success"

    subsampled_gif_path = os.path.join(output_dir, f"{file_base_name}.gif")
    subsampled_states_path = os.path.join(output_dir, f"{file_base_name}_states.npy")

    save_frames_and_states(subsampled_frames, subsampled_gif_path, subsampled_states, subsampled_states_path)
    print(f"Processed {input_gif_path}: saved subsampled GIF and states to {output_dir}")

# Example usage

def mismatched_subsample_gifs_and_states(input_gif_path, output_dir, frame_indices, input_states_path=None):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_gif_path))[0]
    
    frames, states = load_frames_and_states(input_gif_path, input_states_path)
    subsampled_frames = [frames[idx] for idx in frame_indices]
    subsampled_states = [states[idx] for idx in frame_indices]

    subsampled_gif_path = os.path.join(output_dir, f"{base_name}.gif")
    subsampled_states_path = os.path.join(output_dir, f"{base_name}_states.npy")

    save_frames_and_states(subsampled_frames, subsampled_gif_path, subsampled_states, subsampled_states_path)


medium_frame_indices = {
    "window-open-v2": list(range(5)) + list(range(26, 38)) + [49, 50, 51], 
    "button-press-v2": list(range(15)) + [31, 43, 44],
    "door-close-v2": list(range(15)) + [31, 43, 44],
    "door-open-v2":  list(range(15)) + [31, 32, 55, 56],
    "stick-push-v2": list(range(10, 25)) + [49, 50, 51],
    "push-v2": list(range(10, 25)) + [49, 50, 51],
    "door-lock-v2": list(range(20)) + [28,29,30] + [61, 62],
    "lever-pull-v2": list(range(5, 18)) + [31, 32, 55, 56, 60, 61],
    "hand-insert-v2": list(range(5, 18)) + [21, 22, 31, 32, 55, 56],
    "basketball-v2": list(range(7, 20)) + [25, 26, 31, 32, 55, 56]
}

if __name__=="__main__":
    env_name = "metaworld"
    camera_name = "d"
    mismatched = True

    for task_name in medium_frame_indices.keys():
        # default gif path for demos
        input_gif_path = get_demo_gif_path(env_name, task_name, camera_name, demo_num=0, num_frames="d") 
        # new gif path
        output_dir = get_demo_dir(env_name, task_name, camera_name, mismatched=mismatched) 

        if os.path.exists(input_gif_path):
            mismatched_subsample_gifs_and_states(input_gif_path, output_dir, frame_indices=medium_frame_indices[task_name])
            print(f"Successfully created mistmatched demo for {task_name}")
        else:
            print(f"{task_name} does not exist")
            


