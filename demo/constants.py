import os

BASE_DEMO_DIR = 'create_demo'

def get_demo_gif_path(env_name, task_name, camera_name, demo_num, num_frames='d', mismatched=False, random_mismatched_info={}):
    """
    Parameters:
    - random_mismatched_info: If we are getting demo for random mismatched trajectory. Should have the keys:
        {
            'mismatch_level': '1outof5_mismatched', # options: '1outof5_mismatched', '2outof5_mismatched', '3outof5_mismatched'
            'run_num': 0
        }
    """
    if camera_name=="d":
        camera_name = CAMERA[task_name]

    dir_path = get_demo_dir(env_name, task_name, camera_name, num_frames, mismatched, random_mismatched_info)
    return os.path.join(dir_path, f"{task_name}_{camera_name}_{demo_num}.gif")

def get_demo_dir(env_name, task_name, camera_name, num_frames='d', mismatched=False, random_mismatched_info={}, linear_mismatched_info={}):
    """
    Parameters:
    - random_mismatched_info: If we are getting demo for random mismatched trajectory. Should have the keys:
        {
            'speed_type': 'fast', # options: 'fast', 'slow', 'mixed'
            'mismatch_level': '1outof5_mismatched', # options: '1outof5_mismatched', '2outof5_mismatched', '3outof5_mismatched'
            'run_num': 0
        }
    """
    if camera_name=="d":
        camera_name = CAMERA[task_name]

    if mismatched:
        if num_frames == "d":
            return os.path.join(BASE_DEMO_DIR, f"{env_name}_demos/{task_name}/mismatched")
        else:
            return os.path.join(BASE_DEMO_DIR, f"{env_name}_demos/{task_name}/mismatched/subsampled_{num_frames}")
    elif random_mismatched_info != {}:
        speed_type = random_mismatched_info['speed_type']
        mismatch_level = random_mismatched_info['mismatch_level']  # one of '1outof5_mismatched', '2outof5_mismatched', '3outof5_mismatched'
        run_num = int(random_mismatched_info['run_num'])
        return os.path.join(BASE_DEMO_DIR, f"{env_name}_demos/{task_name}/random_mismatched_{speed_type}/{mismatch_level}/{mismatch_level}_{run_num}")
    elif num_frames == "d":
        return os.path.join(BASE_DEMO_DIR, f"{env_name}_demos/{task_name}/default")
    else:
        return os.path.join(BASE_DEMO_DIR, f"{env_name}_demos/{task_name}/frames_{num_frames}")

CAMERA = {
    'button-press-v2': 'corner',
    'door-close-v2': 'corner',
    'hammer-v2': 'corner3',
    'drawer-close-v2': 'corner',
    'drawer-open-v2': 'corner',
    'door-open-v2': 'corner3',
    'bin-picking-v2': 'corner',
    'button-press-topdown-v2': 'corner',
    'door-unlock-v2': 'corner',
    'basketball-v2': 'corner',
    'basketball-v3': 'corner',
    'plate-slide-v2': 'corner',
    'hand-insert-v2': 'corner3', # corner in temporalOT paper
    'peg-insert-side-v2': 'corner3',
    'assembly-v3': 'corner',
    'assembly-v2': 'corner',
    'push-wall-v2': 'corner',
    'soccer-v2': 'corner',
    'disassemble-v2': 'corner',
    'pick-place-wall-v3': 'corner3',
    'pick-place-v2': 'corner3',
    'push-v2': 'corner3',
    'lever-pull-v2': 'corner3', # corner4 in temporalOT paper
    'stick-pull-v2': 'corner3',
    'shelf-place-v2': 'corner',
    'window-close-v2': 'corner3',
    'window-open-v2': 'corner3',
    'reach-v2': 'corner3',
    'button-press-wall-v2': 'corner',
    'box-close-v2': 'corner3',
    'stick-push-v2': 'corner',
    'handle-pull-v2': 'corner3',
    'door-lock-v2': 'corner',
}

MAX_PATH_LENGTH = {
    'button-press-v2': 125,
    'door-close-v2': 125,
    'hammer-v2': 125,
    'drawer-close-v2': 125,
    'drawer-open-v2': 125,
    'door-open-v2': 125,
    'bin-picking-v2': 175,
    'button-press-topdown-v2': 125,
    'door-unlock-v2': 125,
    'basketball-v2': 175,
    'basketball-v3': 175,
    'plate-slide-v2': 125,
    'hand-insert-v2': 125,
    'peg-insert-side-v2': 150,
    'assembly-v3': 175,
    'assembly-v2': 175,
    'push-wall-v2': 175,
    'soccer-v2': 125,
    'disassemble-v2': 125,
    'pick-place-wall-v3': 175,
    'pick-place-v2': 125,
    'push-v2': 125,
    'lever-pull-v2': 175,
    'stick-pull-v2': 175,
    'shelf-place-v2': 175,
    'window-close-v2': 125,
    'window-open-v2': 125,
    'reach-v2': 125,
    'button-press-wall-v2': 125,
    'box-close-v2': 175,
    'stick-push-v2': 125,
    'handle-pull-v2': 175,
    'door-lock-v2': 125,
}
