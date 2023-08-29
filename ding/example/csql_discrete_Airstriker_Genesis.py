import os
import gym
from easydict import EasyDict
from ditk import logging
from ding.envs import setup_ding_env_manager
from ding.model import DQN
from ding.policy import CSQLDiscretePolicy
from ding.envs import DingEnvWrapper, BaseEnvManagerV2
from ding.data import create_dataset
from ding.config import compile_config
from ding.framework import task, ding_init
from ding.framework.context import OfflineRLContext
from ding.framework.middleware import interaction_evaluator, trainer, CkptSaver, offline_data_fetcher, offline_logger, \
    wandb_offline_logger
from ding.utils import set_pkg_seed

from ding.config.CSQLDiscrete.gymretro_airstrikergenesis import cfg, env


def main(cfg):
    # If you don't have offline data, you need to prepare if first and set the data_path in config
    # For demostration, we also can train a RL policy (e.g. SAC) and collect some data
    logging.getLogger().setLevel(logging.INFO)
    cfg = compile_config(cfg, policy=CSQLDiscretePolicy)
    airstrikergenesis_env = env(cfg=cfg.env)
    ding_init(cfg)
    with task.start(async_mode=False, ctx=OfflineRLContext()):
        evaluator_env = setup_ding_env_manager(
            airstrikergenesis_env, env_num=cfg.env.evaluator_env_num, debug=False, caller='evaluator'
        )
        set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
        dataset = create_dataset(cfg)
        model = DQN(**cfg.policy.model)
        policy = CSQLDiscretePolicy(cfg.policy, model=model)
        checkpoint_save_dir = os.path.join(cfg.exp_name, "ckpt")

        task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env, render=True))
        task.use(CkptSaver(policy, save_dir=checkpoint_save_dir, train_freq=100000))
        task.use(offline_data_fetcher(cfg, dataset))
        task.use(trainer(cfg, policy.learn_mode))
        

        task.use(
            wandb_offline_logger(
                cfg=EasyDict(
                    dict(
                        gradient_logger=False,
                        plot_logger=True,
                        video_logger=True,
                        action_logger=False,
                        return_logger=False,
                        vis_dataset=False,
                    )
                ),
                metric_list=policy.monitor_vars(),
                model=policy._model,
                anonymous=True,
                project_name=cfg.exp_name,
                wandb_sweep=False,
            )
        )
        task.use(offline_logger())
        task.run()


if __name__ == "__main__":
    main(cfg)
