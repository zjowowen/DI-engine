from easydict import EasyDict

action_shape = 17
obs_shape =376

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
        stop_value=10000,
        act_scale=True,
        rew_clip=True,
    ),
    policy=dict(
        cuda=True,
        random_collect_size=10000,
        action_space='general',
        model=dict(
            obs_shape=obs_shape,
            action_shape=action_shape,
            twin_critic=True,
            action_space='general',
            customized_model=True,
            actor=dict(
                model_type='GaussianTanh',
                model=dict(
                    mu_model=dict(
                        hidden_sizes=[obs_shape, 512, 512],
                        activation='tanh',
                        output_size=action_shape,
                        dropout=0,
                        layernorm=False,
                        final_activation='tanh',
                        scale=5.0,
                        shrink=0.01,
                    ),
                    cov=dict(
                        dim=action_shape,
                        functional=True,
                        random_init=False,
                        sigma_lambda=dict(
                            hidden_sizes=[obs_shape, 512, 512],
                            activation='tanh',
                            output_size=action_shape,
                            dropout=0,
                            layernorm=False,
                        ),
                        sigma_offdiag=dict(
                            hidden_sizes=[obs_shape, 512, 512],
                            activation='tanh',
                            output_size=int(action_shape*(action_shape-1)//2),
                            dropout=0,
                            layernorm=False,
                        ),
                    ),
                ),
            ),
            critic=dict(
                model_num=2,
                model=dict(
                    hidden_sizes=[obs_shape+action_shape, 512, 512],
                    activation='relu',
                    output_size=1,
                    dropout=0,
                    layernorm=False,
                ),
            ),
        ),
        learn=dict(
            update_per_collect=1,
            batch_size=2048,
            learning_rate_q=3e-4,
            learning_rate_policy=1e-4,
            learning_rate_alpha=0.001,
            ignore_done=False,
            target_theta=0.01,
            discount_factor=0.99,
            alpha=0.2,
            target_entropy=-20,
            auto_alpha=True,
            q_grad_clip=2000,
            policy_grad_clip=0.5,
            weight_decay=0.0001,
        ),
        collect=dict(
            n_sample=1,
            unroll_len=1,
        ),
        command=dict(),
        eval=dict(evaluator=dict(eval_freq=1,),),
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
