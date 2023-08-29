from ding.envs import BaseEnv, DingEnvWrapper
import gym


class AirstrikerGenesis_gameover_wrapper(gym.Wrapper):

    def __init__(self, env, one_live_mode=False):
        super().__init__(env)
        self.one_live_mode = one_live_mode
        self.gameover_counter = 10
        if one_live_mode:
            self.lives = 3

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.one_live_mode:
            if info['lives'] < self.lives:
                done = True
            else:
                self.lives = info['lives']

        if info['gameover'] == 4:
            self.gameover_counter -= 1
            if self.gameover_counter <= 0:
                done = True
        else:
            self.gameover_counter = 10

        return obs, reward, done, info

    def reset(self):
        self.gameover_counter = 10
        if self.one_live_mode:
            self.lives = 3
        return self.env.reset()


def AirstrikerGenesis_action_dtype_transform(action):
    if action == 0:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif action == 1:
        return [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif action == 2:
        return [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    elif action == 3:
        return [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    elif action == 4:
        return [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    elif action == 5:
        return [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    else:
        raise ValueError('Invalid action!!')


def env(cfg, seed_api=True, caller='collector', **kwargs) -> BaseEnv:

    import retro
    from ding.envs.env_wrappers import RewardScaleWrapper, ActionSpaceTransformWrapper, NoopWrapper, \
        WarpFrameWrapper, FrameStackWrapper, TimeLimitWrapper, EvalEpisodeReturnEnv, ScaledFloatFrameWrapper

    env_wrapper = []
    if "time_limit" in cfg and cfg.time_limit > 0:
        env_wrapper.append(lambda env: TimeLimitWrapper(env, max_limit=cfg.time_limit))
    if cfg.env_id == "Airstriker-Genesis":
        env_wrapper.append(
            lambda env: AirstrikerGenesis_gameover_wrapper(env, one_live_mode=True if caller == 'collector' else False)
        )
        env_wrapper.append(
            lambda env: ActionSpaceTransformWrapper(
                env, AirstrikerGenesis_action_dtype_transform, action_space=gym.spaces.Discrete(6)
            )
        )
    if caller == 'collector':
        env_wrapper.append(lambda env: RewardScaleWrapper(env, scale=0.01))
    env_wrapper.append(lambda env: NoopWrapper(env, freq=5, noop_action=0))
    env_wrapper.append(lambda env: WarpFrameWrapper(env, size=84))
    if "change_obs_dtype_and_scale" in cfg and cfg.change_obs_dtype_and_scale:
        env_wrapper.append(lambda env: ScaledFloatFrameWrapper(env))
    env_wrapper.append(lambda env: FrameStackWrapper(env, n_frames=4))
    env_wrapper.append(lambda env: EvalEpisodeReturnEnv(env))

    return DingEnvWrapper(
        env=retro.make(game=cfg.env_id),
        cfg={'env_wrapper': env_wrapper},
        seed_api=seed_api,
        caller=caller,
    )
