import gym
from ditk import logging
from ding.envs import setup_ding_env_manager
from ding.model import QRDQN
from ding.policy import CQLDiscretePolicy
from ding.envs import DingEnvWrapper, BaseEnvManagerV2
from ding.data import create_dataset
from ding.config import compile_config
from ding.framework import task, ding_init
from ding.framework.context import OfflineRLContext
from ding.framework.middleware import interaction_evaluator, trainer, CkptSaver, offline_data_fetcher, offline_logger
from ding.utils import set_pkg_seed

from ding.config.CQLDiscrete.gymretro_airstrikergenesis import cfg, env


def main(cfg):
    # If you don't have offline data, you need to prepare if first and set the data_path in config
    # For demostration, we also can train a RL policy (e.g. SAC) and collect some data
    logging.getLogger().setLevel(logging.INFO)
    cfg = compile_config(cfg, policy=CQLDiscretePolicy)
    airstrikergenesis_env = env(cfg=cfg.env)
    ding_init(cfg)
    with task.start(async_mode=False, ctx=OfflineRLContext()):
        evaluator_env = setup_ding_env_manager(
            airstrikergenesis_env, env_num=cfg.env.evaluator_env_num, debug=True, caller='evaluator'
        )
        set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
        dataset = create_dataset(cfg)
        model = QRDQN(**cfg.policy.model)
        policy = CQLDiscretePolicy(cfg.policy, model=model)

        task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
        task.use(offline_data_fetcher(cfg, dataset))
        task.use(trainer(cfg, policy.learn_mode))
        #task.use(CkptSaver(policy, cfg.exp_name, train_freq=100))
        task.use(offline_logger())
        task.run()


if __name__ == "__main__":
    main(cfg)
