import datetime
import io
import random
import traceback
from collections import defaultdict
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset
import pickle


def episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1


def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open('wb') as f:
            f.write(bs.read())


def load_episode(fn):
    with fn.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode


def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


#################
# Replay Buffer #
#################
class ReplayBufferStorage:

    def __init__(self, data_specs, replay_dir):
        self._data_specs = data_specs
        self._replay_dir = replay_dir
        replay_dir.mkdir(exist_ok=True)
        (replay_dir / 'backup').mkdir(exist_ok=True)
        self._current_episode = defaultdict(list)
        self._preload()

    def __len__(self):
        return self._num_transitions

    def add(self, time_step, early_stop=False):
        for spec in self._data_specs:
            value = time_step[spec.name]
            if np.isscalar(value):
                value = np.full(spec.shape, value, spec.dtype)
        
            assert spec.shape == value.shape and spec.dtype == value.dtype
            self._current_episode[spec.name].append(value)
        if time_step.last() or early_stop:
            episode = dict()
            for spec in self._data_specs:
                value = self._current_episode[spec.name]
                episode[spec.name] = np.array(value, spec.dtype)
            self._current_episode = defaultdict(list)

            self._store_episode(episode)

    def _preload(self):
        self._num_episodes = 0
        self._num_transitions = 0
        for fn in self._replay_dir.glob('*.npz'):
            _, _, eps_len = fn.stem.split('_')
            self._num_episodes += 1
            self._num_transitions += int(eps_len)

    def _store_episode(self, episode):
        eps_idx = self._num_episodes
        eps_len = episode_len(episode)
        self._num_episodes += 1
        self._num_transitions += eps_len
        ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        eps_fn = f'{ts}_{eps_idx}_{eps_len}.npz'
        save_episode(episode, self._replay_dir / eps_fn)

    def update_parameters(self, parameters):
        for k, v in parameters.items():
            self.__dict__[k] = v

        data = {'_discount': self._discount}
        fn = self._replay_dir / 'parameters.pkl'
        with open(str(fn), 'wb') as f:
            pickle.dump(data, f)


class ReplayBuffer(IterableDataset):

    def __init__(self, replay_dir, max_size, num_workers, nstep, discount,
                 fetch_every, save_experiences):
        self._replay_dir = replay_dir
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        self._episodes = dict()
        self._nstep = nstep
        self._discount = discount
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self._save_experiences = save_experiences

    def _sample_episode(self):
        eps_fn = random.choice(self._episode_fns)
        
        return self._episodes[eps_fn]

    def _store_episode(self, eps_fn):
        try:
            episode = load_episode(eps_fn)
        except:
            return False
        eps_len = episode_len(episode)
        while eps_len + self._size > self._max_size:
            early_eps_fn = self._episode_fns.pop(0)
            early_eps = self._episodes.pop(early_eps_fn)
            self._size -= episode_len(early_eps)
            early_eps_fn.unlink(missing_ok=True)
        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()
        self._episodes[eps_fn] = episode
        self._size += eps_len

        if self._save_experiences:
            eps_fn.replace(eps_fn.parent / 'backup' / eps_fn.name)
        else:
            eps_fn.unlink(missing_ok=True)
        return True

    def _try_fetch(self):
        if self._samples_since_last_fetch < self._fetch_every:
            return

        fn = self._replay_dir / 'parameters.pkl'
        if os.path.exists(fn):
            with fn.open('rb') as f:
                parameters = pickle.load(f)
                for k, v in parameters.items():
                    self.__dict__[k] = v

        self._samples_since_last_fetch = 0
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        eps_fns = sorted(self._replay_dir.glob('*.npz'), reverse=True)
        fetched_size = 0
        for eps_fn in eps_fns:
            eps_idx, eps_len = [int(x) for x in eps_fn.stem.split('_')[1:]]
            if eps_idx % self._num_workers != worker_id:
                continue
            if eps_fn in self._episodes.keys():
                break
            if fetched_size + eps_len > self._max_size:
                break
            fetched_size += eps_len
            if not self._store_episode(eps_fn):
                break

    def _sample(self):
        try:
            self._try_fetch()
        except:
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()

        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode) - self._nstep + 1) + 1

        obs = episode['observation'][idx - 1]
        action = episode['action'][idx]
        reward = np.zeros_like(episode['reward'][idx])
        discount = np.ones_like(episode['discount'][idx])
        #print(f"discount: {self._discount()}")
        #print(f"discount count: {self._discount.count}")

        step_rewards = []
        for i in range(self._nstep):
            step_reward = episode['reward'][idx + i]
            step_rewards.append(step_reward)
            # reward += discount * step_reward
            # discount *= episode['discount'][idx + i] * self._discount()
        
        next_obs = episode['observation'][idx + self._nstep - 1]
        step_rewards = np.stack(step_rewards) # convert to numpy first for speed

        return (obs, action, torch.as_tensor(step_rewards), discount, next_obs)

    def __iter__(self):
        while True:
            yield self._sample()

#################
# Expert Buffer #
#################
class ExpertReplayBuffer(IterableDataset):
    def __init__(self, dataset_path, num_demos, obs_type):
        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)
            if obs_type == 'pixels':
                obses, _, actions, _, _ = data
            elif obs_type == 'features':
                _, obses, actions, _, _ = data
        print(f"Loaded obses length = {len(obses)}.")
        self._episodes = []
        for i in range(num_demos):
            episode = dict(observation=obses[i], action=actions[i])
            self._episodes.append(episode)

    def _sample_episode(self):
        episode = random.choice(self._episodes)
        return episode

    def _sample(self):
        episode = self._sample_episode()
        idx = np.random.randint(0, episode_len(episode)) + 1
        obs = episode['observation'][idx]
        action = episode['action'][idx]

        return (obs, action)

    def __iter__(self):
        while True:
            yield self._sample()


#################
# Buffer Loader #
#################
def make_replay_loader(replay_dir, max_size, batch_size, num_workers,
                       save_experiences, nstep, discount):




    max_size_per_worker = max_size // max(1, num_workers)

    iterable = ReplayBuffer(replay_dir,
                            max_size_per_worker,
                            num_workers,
                            nstep,
                            discount,
                            fetch_every=1000,
                            save_experiences=save_experiences)

    loader = torch.utils.data.DataLoader(iterable,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         pin_memory=True,
                                         worker_init_fn=_worker_init_fn)
    return loader


def make_expert_replay_loader(replay_dir, batch_size, num_demos, obs_type):
    iterable = ExpertReplayBuffer(replay_dir, num_demos, obs_type)
    loader = torch.utils.data.DataLoader(iterable,
                                         batch_size=batch_size,
                                         num_workers=1,
                                         pin_memory=True,
                                         worker_init_fn=_worker_init_fn)
    return loader
