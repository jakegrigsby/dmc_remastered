import os

SUITE_DIR = os.path.dirname(__file__)
DMCR_VARY = [
    "bg",
    "floor",
    "body",
    "target",
    "reflectance",
    "camera",
    "light",
    "body_shape",
    "motors",
    "friction",
]
ALL_ENVS = {}
VISUAL_ENVS = {}
DYNAMICS_ENVS = {}
GOAL_ENVS = {}


def register(
    domain: str, task: str, visuals_vary: bool, dynamics_vary: bool, goals_vary: bool
):
    def _register(func):
        if domain not in ALL_ENVS:
            ALL_ENVS[domain] = {}
        ALL_ENVS[domain][task] = func
        if visuals_vary:
            if domain not in VISUAL_ENVS:
                VISUAL_ENVS[domain] = {}
            VISUAL_ENVS[domain][task] = func
        if dynamics_vary:
            if domain not in DYNAMICS_ENVS:
                DYNAMICS_ENVS[domain] = {}
            DYNAMICS_ENVS[domain][task] = func
        if goals_vary:
            if domain not in GOAL_ENVS:
                GOAL_ENVS[domain] = {}
            GOAL_ENVS[domain][task] = func
        return func

    return _register


# register all the tasks
from .ball_in_cup import catch
from .benchmarks import (
    visual_classic,
    visual_generalization,
    dynamics_generalization,
    full_generalization,
    goal_generalization,
)
from .cartpole import balance, balance_sparse, swingup, swingup_sparse
from .cheetah import run
from .finger import spin, turn_easy, turn_hard
from .fish import swim, upright
from .generate_visuals import get_assets
from .hopper import hop, stand
from .humanoid import run, stand, walk
from .pendulum import swingup
from .reacher import easy, hard
from .walker import get_model, run, stand, walk
from .wrapper import make
