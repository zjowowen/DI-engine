from ding.envs import BaseEnv, DingEnvWrapper

def env(cfg, seed_api=True, caller='collector', **kwargs) -> BaseEnv:

    import retro
    import gym
    from ding.envs.env_wrappers import RewardScaleWrapper, ActionSpaceTransformWrapper, NoopWrapper, \
        WarpFrameWrapper, FrameStackWrapper, TimeLimitWrapper, EvalEpisodeReturnEnv, ScaledFloatFrameWrapper
    
    env_wrapper=[]
    if "time_limit" in cfg and cfg.time_limit>0:
        env_wrapper.append(lambda env: TimeLimitWrapper(env, max_limit=cfg.time_limit))
    if cfg.env_id == "Airstriker-Genesis":
        def AirstrikerGenesis_action_dtype_transform(action):
            if action == 0:
                return [0,0,0,0,0,0,0,0,0,0,0,0]
            elif action == 1:
                return [1,0,0,0,0,0,0,0,0,0,0,0]
            elif action == 2:
                return [0,0,1,0,0,0,0,0,0,0,0,0]
            elif action == 3:
                return [0,0,0,1,0,0,0,0,0,0,0,0]
            elif action == 4:
                return [1,0,0,0,0,0,1,0,0,0,0,0]
            elif action == 5:
                return [1,0,0,0,0,0,0,1,0,0,0,0]
            else:
                raise ValueError('Invalid action!!')
        env_wrapper.append(lambda env: ActionSpaceTransformWrapper(env, AirstrikerGenesis_action_dtype_transform, action_space=gym.spaces.Discrete(6)))
    env_wrapper.append(lambda env: RewardScaleWrapper(env, scale=0.01))
    env_wrapper.append(lambda env: NoopWrapper(env, freq=5, noop_action=0))
    env_wrapper.append(lambda env: WarpFrameWrapper(env, size=160))
    if "change_obs_dtype_and_scale" in cfg and cfg.change_obs_dtype_and_scale:
        env_wrapper.append(lambda env: ScaledFloatFrameWrapper(env))
    env_wrapper.append(lambda env: FrameStackWrapper(env, n_frames=4))
    env_wrapper.append(lambda env: EvalEpisodeReturnEnv(env))

    return DingEnvWrapper(
        env=retro.make(game=cfg.env_id),
        cfg={
            'env_wrapper': env_wrapper
        },
        seed_api=seed_api,
        caller=caller,
    )
