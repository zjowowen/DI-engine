from easydict import EasyDict

change_obs_dtype_and_scale=True

cfg = dict(
    exp_name='Airstriker-Genesis-DQN',
    seed=0,
    env=dict(
        env_id='Airstriker-Genesis',
        collector_env_num=8,
        evaluator_env_num=4,
        n_evaluator_episode=4,
        fram_stack=4,
        stop_value=30000,
        time_limit=10000,
        change_obs_dtype_and_scale=change_obs_dtype_and_scale,
    ),
    policy=dict(
        cuda=True,
        random_collect_size=25000,
        priority=False,
        discount_factor=0.99,
        nstep=3,
        learn=dict(
            update_per_collect=10,
            batch_size=32,
            learning_rate=0.0001,
            # Frequency of target network update.
            target_update_freq=500,
            change_obs_dtype_and_scale=change_obs_dtype_and_scale,
        ),
        model=dict(
            obs_shape=[4, 84, 84],
            action_shape= 6,
            encoder_hidden_size_list=[128, 128, 512],
        ),
        collect=dict(n_sample=100, ),
        eval=dict(render=True),
        other=dict(
            eps=dict(
                type='exp',
                start=1.,
                end=0.05,
                decay=10000000,
            ),
            replay_buffer=dict(replay_buffer_size=400000, )
        ),
    ),
    wandb_logger=dict(
        gradient_logger=False,
        video_logger=False,
        plot_logger=True,
        action_logger=False,
        return_logger=False,
    ),
)

cfg = EasyDict(cfg)

import ding.envs.gym_retro
env = ding.envs.gym_retro.env
