import collections
import random

import gymnasium as gym
import numpy as np
import cv2
from dm_env import specs
from gymnasium import Wrapper, core, spaces

from dmc_remastered import DMCR_VARY


class FrameStack(Wrapper):
    def __init__(self, env, num_stack):
        super().__init__(env)
        self._k = num_stack
        self._frames = collections.deque([], maxlen=num_stack)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=np.repeat(env.observation_space.low, num_stack, axis=0),
            high=np.repeat(env.observation_space.high, num_stack, axis=0),
            shape=((shp[0] * num_stack,) + shp[1:]),
            dtype=env.observation_space.dtype,
        )

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)


def _spec_to_box(spec):
    def extract_min_max(s):
        assert s.dtype == np.float64 or s.dtype == np.float32
        dim = int(np.prod(s.shape))
        if type(s) == specs.Array:
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        elif type(s) == specs.BoundedArray:
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0)
    high = np.concatenate(maxs, axis=0)
    assert low.shape == high.shape
    return spaces.Box(low, high, dtype=np.float32)


def _flatten_obs(obs):
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0)


class OldGymWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset()
        return obs

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        return state, reward, (terminated or truncated), info


class DMC_Remastered_Env(core.Env):
    """
    A gym-like wrapper for DeepMind Control, that uses a list
    of visual and dynamics seeds to create a new env each reset

    source: https://github.com/denisyarats/dmc2gym
    """

    def __init__(
        self,
        task_builder,
        visual_seed_generator,
        dynamics_seed_generator,
        height=84,
        width=84,
        camera_id=0,
        frame_skip=1,
        from_pixels=True,
        environment_kwargs=None,
        channels_first=True,
        vary=DMCR_VARY,
    ):
        self._height = height
        self._width = width
        self._camera_id = camera_id
        self._frame_skip = frame_skip
        self._channels_first = channels_first

        self._dynamics_seed_gen = dynamics_seed_generator
        self._visual_seed_gen = visual_seed_generator
        self._task_builder = task_builder

        self._env = self._task_builder(dynamics_seed=0, visual_seed=0, vary=vary)
        self._vary = vary

        # true and normalized action spaces
        self._true_action_space = _spec_to_box([self._env.action_spec()])
        self._norm_action_space = spaces.Box(
            low=-1.0, high=1.0, shape=self._true_action_space.shape, dtype=np.float32
        )
        # create observation space
        shape = [3, height, width] if channels_first else [height, width, 3]
        self._observation_space = spaces.Box(
            low=0, high=255, shape=shape, dtype=np.uint8
        )

        self._state_space = _spec_to_box(self._env.observation_spec().values())

        self._from_pixels = from_pixels

        self.current_state = None

        self.make_new_env()

    def make_new_env(self):
        dynamics_seed = self._dynamics_seed_gen()
        visual_seed = self._visual_seed_gen()
        # print(f"dynamics seed {dynamics_seed}")
        # print(f"visual seed {visual_seed}")
        self._env = self._task_builder(
            dynamics_seed=dynamics_seed,
            visual_seed=visual_seed,
            vary=self._vary,
        )
        self.seed(seed=dynamics_seed)

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _get_obs(self, time_step):
        if self._from_pixels:
            self._current_obs = self.render_env(
                height=self._height, width=self._width, camera_id=self._camera_id
            )
            if self._channels_first:
                self._current_obs = self._current_obs.transpose(2, 0, 1).copy()
            return self._current_obs
        else:
            return self.current_state

    def _convert_action(self, action):
        action = action.astype(np.float64)
        true_delta = self._true_action_space.high - self._true_action_space.low
        norm_delta = self._norm_action_space.high - self._norm_action_space.low
        action = (action - self._norm_action_space.low) / norm_delta
        action = action * true_delta + self._true_action_space.low
        action = action.astype(np.float32)
        return action

    @property
    def observation_space(self):
        if self._from_pixels:
            return self._observation_space
        else:
            return self._state_space

    @property
    def state_space(self):
        return self._state_space

    @property
    def action_space(self):
        return self._norm_action_space

    def seed(self, seed):
        self._true_action_space.seed(seed)
        self._norm_action_space.seed(seed)
        self._observation_space.seed(seed)

    def step(self, action):
        assert self._norm_action_space.contains(action)
        action = self._convert_action(action)
        assert self._true_action_space.contains(action)
        reward = 0
        for _ in range(self._frame_skip):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            truncated = time_step.last()
            if truncated:
                break
        obs = self._get_obs(time_step)
        self.current_state = _flatten_obs(time_step.observation)
        return obs, reward, False, truncated, {}

    def reset(self, soft=False, seed=None, options=None):
        assert (
            seed is None
        ), "Random seeding is built into DMCR constructor, this is here for compatability with gym .26+"
        if not soft:
            # make a whole new env (new visual and/or dynamics seeds)
            self.make_new_env()
        time_step = self._env.reset()
        self.current_state = _flatten_obs(time_step.observation)
        obs = self._get_obs(time_step)
        return obs, {}

    def render_env(self, mode="rgb_array", height=None, width=None, camera_id=0):
        assert mode == "rgb_array", "only support rgb_array mode, given %s" % mode
        height = height or self._height
        width = width or self._width
        camera_id = camera_id or self._camera_id
        return self._env.physics.render(height=height, width=width, camera_id=camera_id)

    def render(self, *args, **kwargs):
        if self._from_pixels:
            img = self._current_obs
            if self._channels_first:
                img = img.transpose(1, 2, 0)
        else:
            img = self.render_env()

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow("DMCR", img)
        cv2.waitKey(1)


from dmc_remastered import ALL_ENVS


def make(
    domain_name,
    task_name,
    visual_seed=None,
    dynamics_seed=None,
    frame_stack=3,
    height=84,
    width=84,
    from_pixels=True,
    camera_id=0,
    frame_skip=1,
    channels_last=False,
    vary=DMCR_VARY,
    **_,
):
    if dynamics_seed is None:
        dynamics_seed = random.randint(1, 1_000_000)
    if visual_seed is None:
        visual_seed = random.randint(1, 1_000_000)

    env = DMC_Remastered_Env(
        ALL_ENVS[domain_name][task_name],
        height=height,
        width=width,
        visual_seed_generator=lambda: visual_seed,
        dynamics_seed_generator=lambda: dynamics_seed,
        camera_id=camera_id,
        from_pixels=from_pixels,
        frame_skip=frame_skip,
        channels_first=not channels_last,
        vary=vary,
    )
    env = FrameStack(env, num_stack=frame_stack)
    return env
