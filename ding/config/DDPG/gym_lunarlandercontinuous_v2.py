from easydict import EasyDict

cfg = dict(
    exp_name='LunarLanderContinuous-V2-DDPG',
    seed=0,
    env=dict(
        env_id='LunarLanderContinuous-v2',
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=260,
        act_scale=True,
    ),
    policy=dict(
        cuda=True,
        random_collect_size=0,
        model=dict(
            obs_shape=8,
            action_shape=2,
            twin_critic=True,
            action_space='regression',
        ),
        learn=dict(
            update_per_collect=2,
            batch_size=128,
            learning_rate_actor=0.001,
            learning_rate_critic=0.001,
            ignore_done=False,  # TODO(pu)
            # (int) When critic network updates once, how many times will actor network update.
            # Delayed Policy Updates in original TD3 paper(https://arxiv.org/pdf/1802.09477.pdf).
            # Default 1 for DDPG, 2 for TD3.
            actor_update_freq=1,
            # (bool) Whether to add noise on target network's action.
            # Target Policy Smoothing Regularization in original TD3 paper(https://arxiv.org/pdf/1802.09477.pdf).
            # Default True for TD3, False for DDPG.
            noise=False,
            noise_sigma=0.1,
            noise_range=dict(
                min=-0.5,
                max=0.5,
            ),
        ),
        collect=dict(
            n_sample=48,
            noise_sigma=0.1,
            collector=dict(collect_print_freq=1000, ),
        ),
        eval=dict(evaluator=dict(eval_freq=100, ), ),
        other=dict(replay_buffer=dict(replay_buffer_size=20000, ), ),
    ),
    wandb_logger=dict(
        gradient_logger=True, video_logger=True, plot_logger=True, action_logger=True, return_logger=False
    ),
)

cfg = EasyDict(cfg)

import ding.envs.gym_env
from functools import partial
env = partial(ding.envs.gym_env.env, continuous=True)
