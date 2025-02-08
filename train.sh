#!/bin/bash

# Training Parameters
TASK_NAME="door-close-v2" # ("button-press-v2" "door-close-v2"  "window-open-v2" "stick-push-v2" "lever-pull-v2")
REWARD_FN="orca" 
SEED="r"

# Demo
NUM_DEMOS=1
NUM_FRAMES="d" # d for default
CAMERA_NAME="d" # d for default (defined in env_utils.CAMERA)
MISMATCHED=false # mismatched = true overrides random_mismatched
RANDOM_MISMATCHED=true 

# Only used if RANDOM_MISMATCHED=true
NUM_SECS=5  
MISMATCHED_LEVEL=1  
SPEED_TYPE='slow' # options are 'slow', 'fast', 'mixed'
RANDOM_MISMATCHED_RUN_NUM=0 

# Reward specific params
TAU=1 # only used by orca
MASK_K=10 # only used by TemporalOT
THRESHOLD=0.9 # only used by the baseline "threshold", which track the progress based on the threshold

INCLUDE_TIMESTEP=true
DISCOUNT_FACTOR=0.9
TRAIN_STEPS=500000

# Use to load a pretrained checkpoint (for example, when training ORCA, we first initialize for 500k with TOT)
USE_CKPT=false
CKPT_PATH=None

# Logging Parameters
WANDB_MODE="disabled"
VIDEO_PERIOD=1200 # N rollouts
EVAL_PERIOD=10000 # N steps
MODEL_PERIOD=100000 # N steps

python main.py \
    train_steps=${TRAIN_STEPS} \
    env_name=${TASK_NAME} \
    reward_fn=${REWARD_FN} \
    mismatched=${MISMATCHED} \
    num_frames=${NUM_FRAMES} \
    random_mismatched=${RANDOM_MISMATCHED} \
    num_secs=${NUM_SECS} \
    mismatched_level=${MISMATCHED_LEVEL} \
    speed_type=${SPEED_TYPE} \
    random_mismatched_run_num=${RANDOM_MISMATCHED_RUN_NUM} \
    obs_type="features" \
    seed=${SEED} \
    discount_factor=${DISCOUNT_FACTOR} \
    num_demos=${NUM_DEMOS} \
    camera_name=${CAMERA_NAME} \
    wandb_mode=${WANDB_MODE} \
    mask_k=${MASK_K} \
    tau=${TAU} \
    threshold=${THRESHOLD} \
    include_timestep=${INCLUDE_TIMESTEP} \
    use_ckpt=${USE_CKPT} \
    ckpt_path=${CKPT_PATH} \
    eval_period=${EVAL_PERIOD} \
    model_period=${MODEL_PERIOD} \
    video_period=${VIDEO_PERIOD} \
    
