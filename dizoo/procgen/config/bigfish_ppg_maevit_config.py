from easydict import EasyDict

bigfish_ppg_default_config = dict(
    exp_name='bigfish_ppg_maevit_seed0',
    env=dict(
        is_train=True,
        env_id='bigfish',
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=40,
        manager=dict(shared_memory=True, ),
    ),
    policy=dict(
        cuda=True,
        model=dict(
            obs_shape=[3, 64, 64],
            action_shape=15,
            encoder_hidden_size_list=[16, 32, 32],
            actor_head_hidden_size=256,
            critic_head_hidden_size=256,
            share_encoder=True,
            maevit_encoder=True,
            encoder_config=dict(
                img_size=64,
                patch_size=16,
                embed_dim=256,
                depth=6,
                num_heads=16,
                decoder_embed_dim=256,
                decoder_depth=6,
                decoder_num_heads=16,
                mlp_ratio=4,
            ),
        ),
        learn=dict(
            learning_rate=0.0005,
            actor_epoch_per_collect=1,
            critic_epoch_per_collect=1,
            value_norm=True,
            batch_size=16384,
            value_weight=0.5,
            entropy_weight=0.01,
            clip_ratio=0.2,
            aux_freq=1,
        ),
        mae_learn=dict(
            learning_rate=0.0005,
            actor_epoch_per_collect=1,
            critic_epoch_per_collect=1,
            batch_size=16384,
            mask_ratio=0.5,
            disable_mae_reconstruct=False,
            stop_trainning_iter=500,
        ),
        collect=dict(n_sample=16384, ),
        eval=dict(evaluator=dict(eval_freq=1000, )),
        other=dict(),
    ),
)
bigfish_ppg_default_config = EasyDict(bigfish_ppg_default_config)
main_config = bigfish_ppg_default_config

bigfish_ppg_create_config = dict(
    env=dict(
        type='procgen',
        import_names=['dizoo.procgen.envs.procgen_env'],
    ),
    env_manager=dict(type='subprocess', ),
    policy=dict(type='ppg_mae'),
)
bigfish_ppg_create_config = EasyDict(bigfish_ppg_create_config)
create_config = bigfish_ppg_create_config

if __name__ == "__main__":

    from ding.entry import serial_pipeline_onpolicy
    serial_pipeline_onpolicy([main_config, create_config], seed=0)
