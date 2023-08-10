import os
from easydict import EasyDict
from tensorboardX import SummaryWriter

from ding.config import compile_config
from ding.worker import BaseLearner, SampleSerialCollector, InteractionSerialEvaluator, AdvancedReplayBuffer
from ding.envs.env_manager.envpool_env_manager import PoolEnvManager
from ding.policy import DQNPolicy
from ding.entry import random_collect
from ding.model import DQN
from ding.utils import set_pkg_seed
from ding.rl_utils import get_epsilon_greedy_fn
from dizoo.atari.config.serial import pong_dqn_envpool_config
import datetime
import wandb
import numpy as np

def main(cfg, collector_env_num=8, collector_batch_size=8, seed=0, max_iterations=int(1e10)):

    cfg.exp_name = 'pong_dqn_envpool' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    cfg.env.collector_env_num=collector_env_num
    cfg.env.collector_batch_size=collector_batch_size
    collector_env_cfg = EasyDict(
        {
            'env_id': cfg.env.env_id,
            'env_num': cfg.env.collector_env_num,
            'batch_size': cfg.env.collector_batch_size,
            # env wrappers
            'episodic_life': True,  # collector: True
            'reward_clip': False,  # collector: True
            'gray_scale': cfg.env.get('gray_scale', True),
            'stack_num': cfg.env.get('stack_num', 4),
        }
    )
    cfg.env["collector_env_cfg"]=collector_env_cfg
    evaluator_env_cfg = EasyDict(
        {
            'env_id': cfg.env.env_id,
            'env_num': cfg.env.evaluator_env_num,
            'batch_size': cfg.env.evaluator_batch_size,
            # env wrappers
            'episodic_life': False,  # evaluator: False
            'reward_clip': False,  # evaluator: False
            'gray_scale': cfg.env.get('gray_scale', True),
            'stack_num': cfg.env.get('stack_num', 4),
        }
    )
    cfg.env["evaluator_env_cfg"]=evaluator_env_cfg
    cfg = compile_config(
        cfg,
        PoolEnvManager,
        DQNPolicy,
        BaseLearner,
        SampleSerialCollector,
        InteractionSerialEvaluator,
        AdvancedReplayBuffer,
        save_cfg=True
    )
    wandb.init(project=cfg.env.env_id, name=cfg.exp_name, config=cfg)
    collector_env = PoolEnvManager(collector_env_cfg)
    evaluator_env = PoolEnvManager(evaluator_env_cfg)
    collector_env.seed(seed)
    evaluator_env.seed(seed)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)

    model = DQN(**cfg.policy.model)
    policy = DQNPolicy(cfg.policy, model=model)
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    collector = SampleSerialCollector(
        cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger, exp_name=cfg.exp_name
    )
    evaluator = InteractionSerialEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )
    replay_buffer = AdvancedReplayBuffer(
        cfg.policy.other.replay_buffer, tb_logger, exp_name=cfg.exp_name, instance_name='replay_buffer'
    )
    eps_cfg = cfg.policy.other.eps
    epsilon_greedy = get_epsilon_greedy_fn(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)

    import time
    start_time = time.time()

    if cfg.policy.random_collect_size > 0:
        collect_kwargs = {'eps': epsilon_greedy(collector.envstep)}
        random_collect(cfg.policy, policy, collector, collector_env, {}, replay_buffer, collect_kwargs=collect_kwargs)
    while collector.envstep<max_iterations:
        info_for_logging = {}
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            info_for_logging['eval_reward_mean'] = np.array(reward['eval_episode_return']).mean()
            info_for_logging['eval_reward_std'] = np.array(reward['eval_episode_return']).std()
            info_for_logging['eval_reward_min'] = np.array(reward['eval_episode_return']).min()
            info_for_logging['eval_reward_max'] = np.array(reward['eval_episode_return']).max()
            if stop:
                break
        eps = epsilon_greedy(collector.envstep)
        new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs={'eps': eps})
        info_for_logging['envstep'] = collector.envstep
        replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
        for i in range(cfg.policy.learn.update_per_collect):
            batch_size = learner.policy.get_attribute('batch_size')
            train_data = replay_buffer.sample(batch_size, learner.train_iter)
            if train_data is not None:
                learner.train(train_data, collector.envstep)
                info_for_logging['train_iter'] = learner.train_iter
        info_for_logging['time']=time.time()-start_time
        wandb.log(data=info_for_logging, step=collector.envstep)

    print(time.time()-start_time)

if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser()
    #parse seed by "--seed"
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--collector_env_num", type=int, default=int(8))
    parser.add_argument("--collector_batch_size", type=int, default=int(8))
    args = parser.parse_args()

    main(EasyDict(pong_dqn_envpool_config), collector_env_num=args.collector_env_num, collector_batch_size=args.collector_batch_size, seed=args.seed, max_iterations=10000000)
