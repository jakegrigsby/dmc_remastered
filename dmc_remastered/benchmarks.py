import random

import numpy as np

from dmc_remastered import ALL_ENVS, GOAL_ENVS, VISUAL_ENVS, DYNAMICS_ENVS, DMCR_VARY

from .wrapper import DMC_Remastered_Env, FrameStack


def uniform_seed_generator(low, high):
    def _generate():
        return random.randint(low, high)

    return _generate


def fixed_seed_generator(seed):
    return lambda: seed


def visual_sim2real(
    domain,
    task,
    num_levels,
    frame_stack=3,
    height=84,
    width=84,
    frame_skip=1,
    channels_last=False,
    vary=DMCR_VARY,
):
    try:
        VISUAL_ENVS[domain][task]
    except KeyError:
        raise KeyError(f"{domain} {task} is not configured for visual generalization.")

    random_start = random.randint(1, 1_000_000)
    train_env = DMC_Remastered_Env(
        task_builder=VISUAL_ENVS[domain][task],
        visual_seed_generator=uniform_seed_generator(
            random_start, random_start + num_levels
        ),
        goal_seed_generator=fixed_seed_generator(0),
        dynamics_seed_generator=fixed_seed_generator(0),
        height=height,
        width=width,
        from_pixels=True,
        camera_id=0,
        frame_skip=frame_skip,
        channels_first=not channels_last,
        vary=vary,
    )
    test_env = DMC_Remastered_Env(
        task_builder=VISUAL_ENVS[domain][task],
        visual_seed_generator=fixed_seed_generator(0),
        dynamics_seed_generator=fixed_seed_generator(0),
        goal_seed_generator=fixed_seed_generator(0),
        height=height,
        width=width,
        from_pixels=True,
        camera_id=0,
        frame_skip=frame_skip,
        channels_first=not channels_last,
        vary=vary,
    )
    train_env = FrameStack(train_env, frame_stack)
    test_env = FrameStack(test_env, frame_stack)
    return train_env, test_env


def visual_classic(
    domain,
    task,
    visual_seed=0,
    frame_stack=3,
    height=84,
    width=84,
    frame_skip=1,
    channels_last=False,
    vary=DMCR_VARY,
):
    train_env = DMC_Remastered_Env(
        task_builder=ALL_ENVS[domain][task],
        visual_seed_generator=fixed_seed_generator(visual_seed),
        dynamics_seed_generator=fixed_seed_generator(0),
        goal_seed_generator=fixed_seed_generator(0),
        height=height,
        width=width,
        from_pixels=True,
        camera_id=0,
        frame_skip=frame_skip,
        channels_first=not channels_last,
        vary=vary,
    )
    test_env = DMC_Remastered_Env(
        task_builder=ALL_ENVS[domain][task],
        visual_seed_generator=fixed_seed_generator(visual_seed),
        dynamics_seed_generator=fixed_seed_generator(0),
        goal_seed_generator=fixed_seed_generator(0),
        height=height,
        width=width,
        from_pixels=True,
        camera_id=0,
        frame_skip=frame_skip,
        channels_first=not channels_last,
        vary=vary,
    )
    train_env = FrameStack(train_env, frame_stack)
    test_env = FrameStack(test_env, frame_stack)
    return train_env, test_env


def goal_generalization(
    domain,
    task,
    num_levels,
    # visual seed only impacts the render
    visual_seed=0,
    vary=DMCR_VARY,
):
    try:
        GOAL_ENVS[domain][task]
    except KeyError:
        raise KeyError(f"{domain} {task} is not configured for goal generalization.")

    random_start = random.randint(1, 1_000_000)
    train_env = DMC_Remastered_Env(
        task_builder=GOAL_ENVS[domain][task],
        dynamics_seed_generator=fixed_seed_generator(0),
        visual_seed_generator=fixed_seed_generator(visual_seed),
        goal_seed_generator=uniform_seed_generator(
            random_start, random_start + num_levels
        ),
        from_pixels=False,
        height=256,
        width=256,
        frame_skip=1,
        vary=vary,
    )
    test_env = DMC_Remastered_Env(
        task_builder=GOAL_ENVS[domain][task],
        dynamics_seed_generator=fixed_seed_generator(0),
        visual_seed_generator=fixed_seed_generator(visual_seed),
        goal_seed_generator=uniform_seed_generator(1, 1_000_000),
        height=256,
        width=256,
        from_pixels=False,
        frame_skip=1,
        vary=vary,
    )
    return train_env, test_env


def dynamics_generalization(
    domain,
    task,
    num_levels,
    # visual seed only impacts the render
    visual_seed=0,
    vary=DMCR_VARY,
):
    try:
        DYNAMICS_ENVS[domain][task]
    except KeyError:
        raise KeyError(
            f"{domain} {task} is not configured for dynamics generalization."
        )

    random_start = random.randint(1, 1_000_000)
    train_env = DMC_Remastered_Env(
        task_builder=DYNAMICS_ENVS[domain][task],
        dynamics_seed_generator=uniform_seed_generator(
            random_start, random_start + num_levels
        ),
        visual_seed_generator=fixed_seed_generator(visual_seed),
        goal_seed_generator=fixed_seed_generator(0),
        from_pixels=False,
        height=256,
        width=256,
        frame_skip=1,
        vary=vary,
    )
    test_env = DMC_Remastered_Env(
        task_builder=DYNAMICS_ENVS[domain][task],
        dynamics_seed_generator=uniform_seed_generator(1, 1_000_000),
        visual_seed_generator=fixed_seed_generator(visual_seed),
        goal_seed_generator=fixed_seed_generator(0),
        height=256,
        width=256,
        from_pixels=False,
        frame_skip=1,
        vary=vary,
    )
    return train_env, test_env


def full_generalization(
    domain,
    task,
    num_levels,
    frame_stack=3,
    height=84,
    width=84,
    frame_skip=1,
    channels_last=False,
    vary=DMCR_VARY,
):
    try:
        VISUAL_ENVS[domain][task]
        DYNAMICS_ENVS[domain][task]
    except KeyError:
        raise KeyError(f"{domain} {task} is not configured for full generalization.")

    random_start = random.randint(1, 1_000_000)
    train_env = DMC_Remastered_Env(
        task_builder=ALL_ENVS[domain][task],
        # note: each should probably have sqrt(num_levels) different seeds
        visual_seed_generator=uniform_seed_generator(
            random_start, random_start + num_levels
        ),
        dynamics_seed_generator=uniform_seed_generator(
            random_start, random_start + num_levels
        ),
        goal_seed_generator=fixed_seed_generator(0),
        height=height,
        width=width,
        from_pixels=True,
        camera_id=0,
        frame_skip=frame_skip,
        channels_first=not channels_last,
        vary=vary,
    )
    test_env = DMC_Remastered_Env(
        task_builder=ALL_ENVS[domain][task],
        visual_seed_generator=uniform_seed_generator(1, 1_000_000),
        dynamics_seed_generator=uniform_seed_generator(1, 1_000_000),
        goal_seed_generator=fixed_seed_generator(0),
        height=height,
        width=width,
        from_pixels=True,
        camera_id=0,
        frame_skip=frame_skip,
        channels_first=not channels_last,
        vary=vary,
    )
    train_env = FrameStack(train_env, frame_stack)
    test_env = FrameStack(test_env, frame_stack)
    return train_env, test_env


def visual_generalization(
    domain,
    task,
    num_levels,
    frame_stack=3,
    height=84,
    width=84,
    frame_skip=1,
    channels_last=False,
    vary=DMCR_VARY,
):
    try:
        VISUAL_ENVS[domain][task]
    except KeyError:
        raise KeyError(f"{domain} {task} is not configured for visual generalization.")

    random_start = random.randint(1, 1_000_000)
    train_env = DMC_Remastered_Env(
        task_builder=VISUAL_ENVS[domain][task],
        visual_seed_generator=uniform_seed_generator(
            random_start, random_start + num_levels
        ),
        dynamics_seed_generator=fixed_seed_generator(0),
        goal_seed_generator=fixed_seed_generator(0),
        height=height,
        width=width,
        from_pixels=True,
        camera_id=0,
        frame_skip=frame_skip,
        channels_first=not channels_last,
        vary=vary,
    )
    test_env = DMC_Remastered_Env(
        task_builder=VISUAL_ENVS[domain][task],
        visual_seed_generator=uniform_seed_generator(1, 1_000_000),
        dynamics_seed_generator=fixed_seed_generator(0),
        goal_seed_generator=fixed_seed_generator(0),
        height=height,
        width=width,
        from_pixels=True,
        camera_id=0,
        frame_skip=frame_skip,
        channels_first=not channels_last,
        vary=vary,
    )
    train_env = FrameStack(train_env, frame_stack)
    test_env = FrameStack(test_env, frame_stack)
    return train_env, test_env
