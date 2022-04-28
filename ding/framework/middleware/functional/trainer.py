from typing import TYPE_CHECKING, Callable, Union
from easydict import EasyDict
import logging
import numpy as np
from ding.policy import Policy
from ding.framework import task
from ding.framework.parallel import Parallel
from ding.utils import data_analyzer

if TYPE_CHECKING:
    from ding.framework import OnlineRLContext, OfflineRLContext


def trainer(cfg: EasyDict, policy: Policy) -> Callable:

    data_analyzer.config(router=Parallel())

    def _train(ctx: Union["OnlineRLContext", "OfflineRLContext"]):

        if ctx.train_data is None:  # no enough data from data fetcher
            return
        train_output = policy.forward(ctx.train_data)
        if ctx.train_iter % cfg.policy.learn.learner.hook.log_show_after_iter == 0:
            logging.info(
                'Current Training: Train Iter({})\tLoss({:.3f})'.format(ctx.train_iter, train_output['total_loss'])
            )
        data_analyzer.record(
            {
                "trainer_train_iter": str(ctx.train_iter),
                "trainer_total_loss": train_output['total_loss']
            }
        )
        ctx.train_iter += 1
        ctx.train_output = train_output

    return _train


def multistep_trainer(cfg: EasyDict, policy: Policy) -> Callable:

    data_analyzer.config(router=Parallel())

    def _train(ctx: Union["OnlineRLContext", "OfflineRLContext"]):

        if ctx.train_data is None:  # no enough data from data fetcher
            return
        train_output = policy.forward(ctx.train_data)
        if ctx.train_iter % cfg.policy.learn.learner.hook.log_show_after_iter == 0:
            loss = np.mean([o['total_loss'] for o in train_output])
            logging.info('Current Training: Train Iter({})\tLoss({:.3f})'.format(ctx.train_iter, loss))
            data_analyzer.record({"trainer_train_iter": str(ctx.train_iter), "trainer_loss": loss.item()})
        ctx.train_iter += len(train_output)
        ctx.train_output = train_output

    return _train


# TODO reward model
