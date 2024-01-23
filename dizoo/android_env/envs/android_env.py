from typing import Any, List, Union, Sequence, Optional
import copy
import numpy as np
import gym

from android_env import loader
import numpy as np
from dm_env import specs


from ding.envs import BaseEnv, BaseEnvTimestep, update_shape
from ding.utils import ENV_REGISTRY
from ding.torch_utils import to_tensor, to_ndarray, to_list
from ding.envs import ObsPlusPrevActRewWrapper


def transfer_dm_env_spec_to_gym_space(dm_env_spec):
    """
    Overview:
        Transfer dm_env spec to gym space.
    """

    gym_space = {}
    for k, v in dm_env_spec.items():
        if isinstance(v, specs.DiscreteArray):
            gym_space[k] = gym.spaces.Discrete(v.num_values)
        elif isinstance(v, specs.BoundedArray):
            gym_space[k] = gym.spaces.Box(
                low=v.minimum if v.minimum.shape==v.shape else np.tile(v.minimum, v.shape),
                high=v.maximum if v.maximum.shape==v.shape else np.tile(v.maximum, v.shape),
                shape=v.shape,
                dtype=v.dtype)
        elif isinstance(v, specs.Array):
            gym_space[k] = gym.spaces.Box(
                low=-np.inf if len(v.shape)==0 else np.tile(-np.inf, v.shape),
                high=np.inf if len(v.shape)==0 else np.tile(np.inf, v.shape),
                shape=v.shape,
                dtype=v.dtype)
        else:
            raise NotImplementedError('Unknown dm_env spec type: {}'.format(type(v)))

    return gym.spaces.Dict(gym_space)
    



@ENV_REGISTRY.register("android_env")
class AndroidEnv(BaseEnv):
    """
    Overview:
        Gym environment class for environments on Android devices.
        (https://github.com/google-deepmind/android_env)
    Interfaces:
        ``__init__``, ``reset``, ``close``, ``seed``, ``step``, \
        ``random_action``, ``enable_save_replay``
    
    .. note::
        Please follow the installation guide of android_env to install the environment.
        Install Android Studio and Android SDK first.
        Then install android_env by:
        .. code-block:: bash
            pip install android_env
        Create an AVD in Android Studio, and find out AVD name, AVD home, SDK root,
        which will be used in the config file.
        The recommeded Android version is 9.0 (API 28), which depends on the game you want to play.
        Then you need to download the game you want to play, 
        https://github.com/google-deepmind/android_env/blob/main/docs/example_tasks.md
        Find out the task path, such as ``~/android_game/apple_flinger/apple_flinger_M_1_1.textproto``.
        Copy the apk file to the LOCAL directory of the python script,
        such as ``~/2020.08.21-apple-flinger-debug.apk``.
    """
    def __init__(self, cfg: dict) -> None:
        
        assert "avd_name" in cfg, "avd_name must be specified in env config"
        assert "android_avd_home" in cfg, "android_avd_home must be specified in env config"
        assert "android_sdk_root" in cfg, "android_sdk_root must be specified in env config"
        assert "task_path" in cfg, "task_path must be specified in env config"

        # default config
        self._cfg = dict(
            avd_name='my_avd',
            android_avd_home='~/.android/avd',
            android_sdk_root='~/Android/Sdk',
            emulator_path='~/Android/Sdk/emulator/emulator', # optional
            adb_path='~/Android/Sdk/platform-tools/adb', # optional
            task_path='./apple_flinger_M_1_1.textproto',
            run_headless=True,
        )

        for k, v in cfg.items():
            self._cfg[k] = v

        self._env = loader.load(
            avd_name=self._cfg["avd_name"],
            android_avd_home=self._cfg["android_avd_home"],
            android_sdk_root=self._cfg["android_sdk_root"],
            emulator_path=self._cfg["emulator_path"],
            adb_path=self._cfg["adb_path"],
            task_path=self._cfg["task_path"],
            run_headless=self._cfg["run_headless"],
        )


        self._action_spec = self._env.action_spec()
        self._action_space = transfer_dm_env_spec_to_gym_space(self._action_spec)
        self._observation_spec = self._env.observation_spec()
        self._observation_space = transfer_dm_env_spec_to_gym_space(self._observation_spec)
        self._reward_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, ), dtype=np.float32)

        self._env.close()


    def reset(self) -> dict[str, np.ndarray]:
        if self._env._is_closed:
            self._env = loader.load(
                avd_name=self._cfg["avd_name"],
                android_avd_home=self._cfg["android_avd_home"],
                android_sdk_root=self._cfg["android_sdk_root"],
                emulator_path=self._cfg["emulator_path"],
                adb_path=self._cfg["adb_path"],
                task_path=self._cfg["task_path"],
                run_headless=self._cfg["run_headless"],
            )
        timestep=self._env.reset()
        return timestep.observation


    def close(self) -> None:
        self._env.close()

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        # no seed option for android_env
        return

    def step(self, action: dict[str, np.ndarray]) -> BaseEnvTimestep:
        timestep = self._env.step(action=action)
        obs = timestep.observation
        rew = timestep.reward
        done = timestep.last()
        info = {}
        return BaseEnvTimestep(obs, rew, done, info)

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        pass

    def random_action(self) -> dict[str, np.ndarray]:
        """Returns a random AndroidEnv action."""
        action = {}
        for k, v in self._action_spec.items():
            if isinstance(v, specs.DiscreteArray):
                action[k] = np.random.randint(low=0, high=v.num_values, dtype=v.dtype)
            else:
                action[k] = np.random.random(size=v.shape).astype(v.dtype)
        return action

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space

    def __repr__(self) -> str:
        return "DI-engine Android Env({})".format(self._cfg.env_id)

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_env_num = cfg.pop('collector_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.is_train = True
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop('evaluator_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.is_train = False
        return [cfg for _ in range(evaluator_env_num)]
