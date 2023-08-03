from easydict import EasyDict

spaceinvaders_dqn_envpool_config = dict(
    exp_name='spaceinvaders_dqn_envpool_seed0',
    env=dict(
        collector_env_num=8,
        collector_batch_size=8,
        evaluator_env_num=8,
        evaluator_batch_size=8,
        n_evaluator_episode=8,
        stop_value=20,
        env_id='SpaceInvaders-v5',
        #'ALE/SpaceInvaders-v5' is available. But special setting is needed after gym make.
        frame_stack=4,
    ),
    policy=dict(
        cuda=True,
        priority=False,
        random_collect_size=5000,
        model=dict(
            obs_shape=[4, 84, 84],
            action_shape=6,
            encoder_hidden_size_list=[128, 128, 512],
        ),
        nstep=3,
        discount_factor=0.99,
        learn=dict(
            update_per_collect=10,
            batch_size=32,
            learning_rate=0.0001,
            target_update_freq=500,
        ),
        collect=dict(n_sample=100, ),
        eval=dict(evaluator=dict(eval_freq=4000, )),
        other=dict(
            eps=dict(
                type='exp',
                start=1.,
                end=0.05,
                decay=1000000,
            ),
            replay_buffer=dict(replay_buffer_size=400000, ),
        ),
    ),
)
spaceinvaders_dqn_envpool_config = EasyDict(spaceinvaders_dqn_envpool_config)
main_config = spaceinvaders_dqn_envpool_config
spaceinvaders_dqn_envpool_create_config = dict(
    env=dict(
        type='atari',
        import_names=['dizoo.atari.envs.atari_env'],
    ),
    env_manager=dict(type='env_pool'),
    policy=dict(type='dqn'),
    replay_buffer=dict(type='deque'),
)
spaceinvaders_dqn_envpool_create_config = EasyDict(spaceinvaders_dqn_envpool_create_config)
create_config = spaceinvaders_dqn_envpool_create_config

if __name__ == '__main__':
    # or you can enter `ding -m serial -c spaceinvaders_dqn_envpool_config.py -s 0`
    from ding.entry import serial_pipeline
    serial_pipeline((main_config, create_config), seed=0)
