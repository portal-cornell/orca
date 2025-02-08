import os 

WANDB_DIR = "./"

def set_os_vars() -> None:

    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    # Get egl (mujoco) rendering to work on cluster
    os.environ["NVIDIA_VISIBLE_DEVICES"] = "all"
    os.environ["NVIDIA_DRIVER_CAPABILITIES"] = "compute,graphics,utility,video"
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["EGL_PLATFORM"] = "device"
    # Get wandb file (e.g. rendered) gif more accessible
    os.environ["WANDB_DIR"] = WANDB_DIR
    # os.environ["LOGURU_LEVEL"] = "INFO"  # Uncomment this if you don't want logger to print logger.debug stuff
