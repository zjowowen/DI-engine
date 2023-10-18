from easydict import EasyDict
import ding.envs.gym_env

cfg = dict(
    exp_name='QbertNoFrameskip-v4-IMPALA',
    seed=0,
    env=dict(
        env_id='QbertNoFrameskip-v4',
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        fram_stack=4,
        stop_value=30000,
        env_wrapper='atari_default',
    ),
    policy=dict(
        cuda=True,
        # (int) the trajectory length to calculate v-trace target
        unroll_len=16,
        random_collect_size=8000,
        model=dict(
            obs_shape=[4, 84, 84],
            action_shape=6,
            encoder_hidden_size_list=[128, 128, 256, 256],
            critic_head_hidden_size=256,
            critic_head_layer_num=3,
            actor_head_hidden_size=256,
            actor_head_layer_num=3,
        ),
        learn=dict(
            # (int) collect n_sample data, train model update_per_collect times
            # here we follow impala serial pipeline
            update_per_collect=2,  # update_per_collect show be in [1, 10]
            # (int) the number of data for a train iteration
            batch_size=128,
            grad_clip_type='clip_norm',
            clip_value=5,
            learning_rate=0.0006,
            # (float) loss weight of the value network, the weight of policy network is set to 1
            value_weight=0.5,
            # (float) loss weight of the entropy regularization, the weight of policy network is set to 1
            entropy_weight=0.01,
            # (float) discount factor for future reward, defaults int [0, 1]
            discount_factor=0.99,
            # (float) additional discounting parameter
            lambda_=0.95,
            # (float) clip ratio of importance weights
            rho_clip_ratio=1.0,
            # (float) clip ratio of importance weights
            c_clip_ratio=1.0,
            # (float) clip ratio of importance sampling
            rho_pg_clip_ratio=1.0,
        ),
        collect=dict(
            # (int) collect n_sample data, train model n_iteration times
            n_sample=16,
            collector=dict(collect_print_freq=1000, ),
        ),
        eval=dict(evaluator=dict(eval_freq=500, )),
        other=dict(replay_buffer=dict(replay_buffer_size=64000, sliced=True), ),
    ),
    wandb_logger=dict(
        gradient_logger=True, video_logger=True, plot_logger=True, action_logger=True, return_logger=False
    ),
)

cfg = EasyDict(cfg)

env = ding.envs.gym_env.env
