from easydict import EasyDict

cfg = dict(
    exp_name='Humanoid-v4-SAC',
    seed=0,
    env=dict(
        env_id='Humanoid-v4',
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        collector_env_num=1,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=6000,
        act_scale=True,
        rew_clip=True,
    ),
    policy=dict(
        cuda=True,
        random_collect_size=10000,
        model=dict(
            obs_shape=376,
            action_shape=17,
            twin_critic=True,
            action_space='reparameterization',
            actor_head_hidden_size=512,
            critic_head_hidden_size=512,
        ),
        learn=dict(
            update_per_collect=1,
            batch_size=256,
            learning_rate_q=3e-4,
            learning_rate_policy=3e-4,
            learning_rate_alpha=3e-4,
            ignore_done=False,
            target_theta=0.005,
            discount_factor=0.99,
            alpha=0.2,
            reparameterization=True,
            target_entropy=-10,
            auto_alpha=True,
        ),
        collect=dict(
            n_sample=1,
            unroll_len=1,
        ),
        command=dict(),
        eval=dict(),
        other=dict(replay_buffer=dict(replay_buffer_size=1000000, ), ),
    ),
    wandb_logger=dict(
        gradient_logger=True,
        video_logger=True,
        plot_logger=True,
        action_logger=True,
        return_logger=False
    ),
)

cfg = EasyDict(cfg)

import ding.envs.gym_env
env = ding.envs.gym_env.env
