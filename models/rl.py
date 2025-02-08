import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from utils import (weight_init, to_torch, soft_update_params, cosine_distance,
                   TruncatedNormal, RandomShiftsAug)

###############
# CNN Encoder #
###############
class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()
        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35
        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(),
                                     nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(),
                                     nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(),
                                     nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())
        self.apply(weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


################
# Actor-Critic #
################
class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim),
                                   nn.Tanh())
        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))
        self.apply(weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)
        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std
        dist = TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim),
                                   nn.Tanh())
        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1))
        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1))
        self.apply(weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)
        return q1, q2


####################
# TemporalOT Agent #
####################
class DDPGAgent:
    def __init__(self,
                 reward_fn,
                 obs_shape,
                 action_shape,
                 device,
                 lr,
                 feature_dim,
                 hidden_dim,
                 critic_target_tau,
                 stddev,
                 stddev_clip,
                 rew_scale=1,
                 auto_rew_scale_factor=10,
                 env_horizon: int = 88,
                 context_num: int = 3,
                 use_encoder: bool = True,
                 include_timestep: bool = False):

        self.reward_fn = reward_fn
        self.include_timestep = include_timestep

        self.context_num = context_num

        self.device = device
        self.lr = lr
        self.critic_target_tau = critic_target_tau
        self.stddev = stddev
        self.stddev_clip = stddev_clip
        self.update_cnt = 0
        self.use_encoder = use_encoder
        self.rew_scale = rew_scale
        self.auto_rew_scale_factor = auto_rew_scale_factor

        # models 
        if self.use_encoder:
            self.encoder = Encoder(obs_shape).to(device)
            self.encoder_target = Encoder(obs_shape).to(device)
            repr_dim = self.encoder.repr_dim
        else:
            repr_dim = obs_shape[0]
            
        self.actor = Actor(repr_dim,
                           action_shape,
                           feature_dim,
                           hidden_dim).to(device)
        self.critic = Critic(repr_dim,
                             action_shape,
                             feature_dim,
                             hidden_dim).to(device)
        self.critic_target = Critic(repr_dim,
                                    action_shape,
                                    feature_dim,
                                    hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        if self.use_encoder:
            self.encoder_opt = optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training 
        if self.use_encoder:
            self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, expl_noise, eval_mode=False):
        obs = torch.as_tensor(obs, device=self.device)
        if self.use_encoder:
            obs = self.encoder(obs.unsqueeze(0))
        else:
            obs = obs.unsqueeze(0)
        dist = self.actor(obs, expl_noise)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
        return action.detach().cpu().numpy()[0]

    def update_critic(self,
                      obs,
                      action,
                      reward,
                      discount,
                      next_obs):

        with torch.no_grad():
            dist = self.actor(next_obs, self.stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        # optimize encoder and critic
        if self.use_encoder:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step() 
        if self.use_encoder:
            self.encoder_opt.step() 

    def update_actor(self, obs):
        dist = self.actor(obs, self.stddev)
        action = dist.sample(clip=self.stddev_clip)

        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)
        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

    def update(self, replay_iter, gamma, update_actor=True):
        
        self.update_cnt += 1
        if self.update_cnt & 1: return None # do not update every other step (double stepping)

        batch = next(replay_iter)
        obs, action, step_reward, _, next_obs = to_torch(batch, self.device) # DO NOT USE DISCOUNT FROM BUFFER
        discount_factors = torch.pow(gamma, torch.arange(step_reward.shape[1], dtype=step_reward.dtype, device=step_reward.device))
        # Multiply step rewards by discount factors and sum across steps
        reward = torch.sum(step_reward.squeeze() * discount_factors, dim=1).unsqueeze(1)
        discount = torch.ones_like(reward) * gamma # for target Q value update

        # augment 
        if self.use_encoder:
            obs = self.aug(obs.float())
            next_obs = self.aug(next_obs.float())
 
        # encode
        if self.use_encoder:
            obs = self.encoder(obs)
            with torch.no_grad():
                next_obs = self.encoder(next_obs)

        # update critic
        self.update_critic(obs, action, reward, discount, next_obs)

        # update actor
        if update_actor:
            self.update_actor(obs.detach())

        # update critic target
        soft_update_params(self.critic,
                           self.critic_target,
                           self.critic_target_tau)
        return discount[0]  # sanity check

    def init_demos(self, cost_encoder, demos):
        self.cost_encoder = cost_encoder
        self.demos = [self.get_context_observations(demo) for demo in demos]

    def rewarder(self, observations):
        scores_list = list()
        rewards_list = list()
        assignment_list = list()
        cost_matrix_list = list()
        progress_list = list()
        obs = torch.as_tensor(observations).to(self.device)

        with torch.no_grad():
            obs = self.cost_encoder(obs)
        obs = self.get_context_observations(obs)

        for exp in self.demos:
            # context cost matrix
            distance_matrix = 0

            for i in range(self.context_num):
                distance_matrix += cosine_distance(obs[i], exp[i])
            distance_matrix /= self.context_num

            rewards, info = self.reward_fn(distance_matrix.cpu().numpy())
            assignment = info["assignment"]
            rewards = rewards.astype(np.float32)
            rewards = self.rew_scale * rewards

            if "progress" in info:
                progress_list.append(info["progress"])

            scores_list.append(np.sum(rewards))
            rewards_list.append(rewards)
            assignment_list.append(assignment)
            cost_matrix_list.append(distance_matrix.cpu().numpy())

        closest_demo_index = np.argmax(scores_list)

        info = {"cost_matrix": cost_matrix_list[closest_demo_index],
                "assignment": assignment_list[closest_demo_index]}

        # if tracking progress then log it
        if len(progress_list) > 0: 
            info["progress"] = progress_list[closest_demo_index]

        final_rewards = rewards_list[closest_demo_index]

        return final_rewards, info

    def set_reward_scale(self, scale):
        self.rew_scale =  self.auto_rew_scale_factor * scale

    def get_reward_scale(self):
        return self.rew_scale


    def get_context_observations(self, observations):
        L = len(observations)
        idx0 = np.arange(L)
        context_observations = [observations[idx0]]
        for i in range(1, self.context_num):
            idx_i = (idx0 + i).clip(0, L-1)
            context_observations.append(observations[idx_i])
        context_observations = torch.stack(context_observations)
        return context_observations.to(self.device)

    def save_snapshot(self):
        keys_to_save = ["actor",
                        "critic",
                        "actor_opt",
                        "critic_opt"]
        payload = {k: self.__dict__[k] for k in keys_to_save}
        return payload

    def load_snapshot(self, payload):
        for k, v in payload.items():
            self.__dict__[k] = v
        self.critic_target.load_state_dict(
            self.critic.state_dict())
        if self.use_encoder:
            self.encoder_target.load_state_dict(
                self.encoder.state_dict())
            self.encoder_opt = optim.Adam(
                self.encoder.parameters(), lr=self.lr)
            self.encoder_opt.load_state_dict(
                payload["encoder_opt"].state_dict())
        self.actor_opt = optim.Adam(
            self.actor.parameters(), lr=self.lr)
        self.actor_opt.load_state_dict(
            payload["actor_opt"].state_dict())
        self.critic_opt = optim.Adam(
            self.critic.parameters(), lr=self.lr)
        self.critic_opt.load_state_dict(
            payload["critic_opt"].state_dict())

