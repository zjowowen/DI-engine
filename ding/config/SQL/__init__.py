from easydict import EasyDict
from . import gym_lunarlander_v2
from . import gymretro_airstrikergenesis

supported_env_cfg = {
    gym_lunarlander_v2.cfg.env.env_id: gym_lunarlander_v2.cfg,
    gymretro_airstrikergenesis.cfg.env.env_id: gymretro_airstrikergenesis.cfg,
}

supported_env_cfg = EasyDict(supported_env_cfg)

supported_env = {
    gym_lunarlander_v2.cfg.env.env_id: gym_lunarlander_v2.env,
    gymretro_airstrikergenesis.cfg.env.env_id: gymretro_airstrikergenesis.env,
}

supported_env = EasyDict(supported_env)
