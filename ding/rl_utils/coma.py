from collections import namedtuple
import torch
import torch.nn.functional as F
from .td import generalized_lambda_returns

from ding.hpc_rl import hpc_wrapper

coma_data = namedtuple('coma_data', ['logit', 'action', 'q_value', 'target_q_value', 'reward', 'weight'])
coma_loss = namedtuple('coma_loss', ['policy_loss', 'q_value_loss', 'entropy_loss'])


def shape_fn_coma(args, kwargs):
    r"""
    Overview:
        Return shape of coma for hpc
    Returns:
        shape: (T, B, A, N)
    """
    if len(args) <= 0:
        tmp = kwargs['data'].logit.shape
    else:
        tmp = args[0].logit.shape
    return tmp


@hpc_wrapper(
    shape_fn=shape_fn_coma, namedtuple_data=True, include_args=[0, 1, 2], include_kwargs=['data', 'gamma', 'lambda_']
)
def coma_error(data: namedtuple, gamma: float, lambda_: float) -> namedtuple:
    """
    Overview:
        Implementation of COMA
    Arguments:
        - data (:obj:`namedtuple`): coma input data with fieids shown in ``coma_data``
    Returns:
        - coma_loss (:obj:`namedtuple`): the coma loss item, all of them are the differentiable 0-dim tensor
    Shapes:
        - logit (:obj:`torch.FloatTensor`): :math:`(T, B, A, N)`, where B is batch size A is the agent num, and N is \
            action dim
        - action (:obj:`torch.LongTensor`): :math:`(T, B, A)`
        - q_value (:obj:`torch.FloatTensor`): :math:`(T, B, A, N)`
        - target_q_value (:obj:`torch.FloatTensor`): :math:`(T, B, A, N)`
        - reward (:obj:`torch.FloatTensor`): :math:`(T, B)`
        - weight (:obj:`torch.FloatTensor` or :obj:`None`): :math:`(T ,B, A)`
        - policy_loss (:obj:`torch.FloatTensor`): :math:`()`, 0-dim tensor
        - value_loss (:obj:`torch.FloatTensor`): :math:`()`
        - entropy_loss (:obj:`torch.FloatTensor`): :math:`()`
    """
    logit, action, q_value, target_q_value, reward, weight = data
    if weight is None:
        weight = torch.ones_like(action)
    q_taken = torch.gather(q_value, -1, index=action.unsqueeze(-1)).squeeze(-1)
    target_q_taken = torch.gather(target_q_value, -1, index=action.unsqueeze(-1)).squeeze(-1)
    T, B, A = target_q_taken.shape
    reward = reward.unsqueeze(-1).expand_as(target_q_taken).reshape(T, -1)
    target_q_taken = target_q_taken.reshape(T, -1)
    return_ = generalized_lambda_returns(target_q_taken, reward[:-1], gamma, lambda_)
    return_ = return_.reshape(T - 1, B, A)
    q_value_loss = (F.mse_loss(return_, q_taken[:-1], reduction='none') * weight[:-1]).mean()

    dist = torch.distributions.categorical.Categorical(logits=logit)
    logp = dist.log_prob(action)
    baseline = (torch.softmax(logit, dim=-1) * q_value).sum(-1).detach()
    adv = (q_taken - baseline).detach()
    entropy_loss = (dist.entropy() * weight).mean()
    policy_loss = -(logp * adv * weight).mean()
    return coma_loss(policy_loss, q_value_loss, entropy_loss)
