import random
import re
import time
import imageio
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal
from PIL import Image
import io
import matplotlib.pyplot as plt
from .math_utils import interquartile_mean_and_ci, mean_and_std

import cv2

def load_gif_frames(path: str, output_type="torch"):
    """
    output_type is either "torch" or "pil"

    Load the gif at the path into a torch tensor with shape (frames, channel, height, width)
    """
    gif_obj = Image.open(path)
    frames = [gif_obj.seek(frame_index) or gif_obj.convert("RGB") for frame_index in range(gif_obj.n_frames)]
    if output_type == "pil":
        return frames
    
    frames = [cv2.resize(np.array(frame), (224, 224)) for frame in frames]
    
    frames_torch = torch.stack([torch.tensor(frame).permute(2, 0, 1) for frame in frames])
    frames_torch = frames_torch.float()
    return frames_torch

def record_demo(demo, video_dir, fname="demo"):
    frames = []
    L = len(demo)
    for i in range(L):
        frames.append(np.transpose(demo[i][-3:, :, :], (1, 2, 0)))
    imageio.mimsave(f"{video_dir}/{fname}.mp4", frames, fps=60)


def get_logger(fname: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=fname,
                        filemode='w',
                        force=True)
    logger = logging.getLogger()
    return logger


def eval_agent(agent, eval_env, obs_type, episode_num=100):
    """
    Return:
        eval_metrics: dict of evaluation metrics (will be logged to wandb)
        final_observations: np array of shape (episode_num, episode_len, obs_dim)
        final_pixels: np array of shape (episode_num, episode_len, 3, 64, 64)
        total_successes: list of success per episode (episode_num,)
    """
    final_successes = 0
    total_successes = []

    observations = []
    pixels = []
    
    for i in range(episode_num):
        ep_observations = []
        ep_pixels = []
        ep_successes = 0
        time_step = eval_env.reset()
        while not time_step.last():
            ep_successes += time_step.observation["goal_achieved"]
            obs = time_step.observation[obs_type]
            pixel = time_step.observation["pixels_large"]
            
            # save the observations from the first 10 eval episodes
            if i < 10:
                ep_observations.append(obs)
                ep_pixels.append(pixel)

            with torch.no_grad(), eval_mode(agent):
                action = agent.act(obs=obs,
                                   expl_noise=0,
                                   eval_mode=True)
            time_step = eval_env.step(action)
        
        if i < 10:
            observations.append(np.stack(ep_observations))
            pixels.append(np.stack(ep_pixels))
        final_successes += time_step.observation["goal_achieved"]
        total_successes.append(ep_successes)

    total_successes = np.array(total_successes)

    final_success_rate = final_successes / episode_num # final succcess per episode
    total_successes_iqm, total_successes_lower, total_successes_upper = interquartile_mean_and_ci(total_successes)
    total_successes_mean, total_successes_std = mean_and_std(total_successes)
    final_observations = np.stack(observations)
    final_pixels = np.stack(pixels)

    eval_metrics = {"eval/final_success_rate": final_success_rate, 
                    "eval/total_success_mean": total_successes_mean,
                    "eval/total_success_std": total_successes_std,
                    "eval/total_success_iqm": total_successes_iqm, 
                    "eval/total_success_lower": total_successes_lower, 
                    "eval/total_success_upper": total_successes_upper,}

    return eval_metrics, final_observations, final_pixels, total_successes

def plot_train_heatmap(matrix, title):
    # Create a heatmap of the cost matrix with grayscale colormap
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap="gray", aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Reference Trajectory")
    plt.ylabel("Learner Trajectory")

    # Save the plot to a BytesIO buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    plt.close()
    buffer.seek(0)

    img = Image.open(buffer)
    return img

def plot_training_performance(df, out_path):
    plt.plot(df["step"], df["eval/total_success_mean"])
    plt.xlabel("step")
    plt.ylabel("cumulative return")
    plt.title("Cumulative return over Training")
    plt.savefig(out_path)

def get_image(time_step):
    image = time_step.observation["pixels_large"][-3:, ...]
    return np.transpose(image, (1, 2, 0))


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def to_torch(xs, device):
    return tuple(torch.as_tensor(x, device=device) for x in xs)


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class eval_mode:

    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


class TruncatedNormal(pyd.Normal):

    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


class RandomShiftsAug(nn.Module):

    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)

class AdaptiveDiscount:
    def __init__(self, discount):
        self.discount = discount
        self.count = 0
    
    def __call__(self):
        return self.discount

    def set_discount(self, new_discount):
        self.discount = new_discount