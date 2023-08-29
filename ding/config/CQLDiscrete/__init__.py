from easydict import EasyDict
from . import gymretro_airstrikergenesis

supported_env_cfg = {
    gymretro_airstrikergenesis.cfg.env.env_id: gymretro_airstrikergenesis.cfg,
}

supported_env_cfg = EasyDict(supported_env_cfg)

supported_env = {
    gymretro_airstrikergenesis.cfg.env.env_id: gymretro_airstrikergenesis.env,
}

supported_env = EasyDict(supported_env)
