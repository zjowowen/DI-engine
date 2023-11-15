from typing import List, Any, Dict, Callable
import numpy as np
import torch
import numpy as np
import treetensor.torch as ttorch
from ding.utils.data import default_collate
from ding.torch_utils import to_tensor, to_ndarray, unsqueeze, squeeze, to_device
import time


def default_preprocess_learn(
        data: List[Any],
        use_priority_IS_weight: bool = False,
        use_priority: bool = False,
        use_nstep: bool = False,
        ignore_done: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Overview:
        Default data pre-processing in policy's ``_forward_learn`` method, including stacking batch data, preprocess \
        ignore done, nstep and priority IS weight.
    Arguments:
        - data (:obj:`List[Any]`): The list of a training batch samples, each sample is a dict of PyTorch Tensor.
        - use_priority_IS_weight (:obj:`bool`): Whether to use priority IS weight correction, if True, this function \
            will set the weight of each sample to the priority IS weight.
        - use_priority (:obj:`bool`): Whether to use priority, if True, this function will set the priority IS weight.
        - use_nstep (:obj:`bool`): Whether to use nstep TD error, if True, this function will reshape the reward.
        - ignore_done (:obj:`bool`): Whether to ignore done, if True, this function will set the done to 0.
    Returns:
        - data (:obj:`Dict[str, torch.Tensor]`): The preprocessed dict data whose values can be directly used for \
            the following model forward and loss computation.
    """
    # data preprocess
    if data[0]['action'].dtype in [np.int64, torch.int64]:
        data = default_collate(data, cat_1dim=True)  # for discrete action
    else:
        data = default_collate(data, cat_1dim=False)  # for continuous action
    if 'value' in data and data['value'].dim() == 2 and data['value'].shape[1] == 1:
        data['value'] = data['value'].squeeze(-1)
    if 'adv' in data and data['adv'].dim() == 2 and data['adv'].shape[1] == 1:
        data['adv'] = data['adv'].squeeze(-1)

    if ignore_done:
        data['done'] = torch.zeros_like(data['done']).float()
    else:
        data['done'] = data['done'].float()

    if data['done'].dim() == 2 and data['done'].shape[1] == 1:
        data['done'] = data['done'].squeeze(-1)

    if use_priority_IS_weight:
        assert use_priority, "Use IS Weight correction, but Priority is not used."
    if use_priority and use_priority_IS_weight:
        if 'priority_IS' in data:
            data['weight'] = data['priority_IS']
        else:  # for compability
            data['weight'] = data['IS']
    else:
        data['weight'] = data.get('weight', None)
    if use_nstep:
        # reward reshaping for n-step
        reward = data['reward']
        reward = torch.stack(reward)
        if len(reward.shape) == 1:
            reward = reward.unsqueeze(1)
        # reward: (batch_size, nstep) -> (nstep, batch_size)
        # data['reward'] = reward.permute(1, 0).contiguous()
        data['reward'] = reward
    else:
        if data['reward'].dim() == 2 and data['reward'].shape[1] == 1:
            data['reward'] = data['reward'].squeeze(-1)

    return data


def fast_preprocess_learn(
        data: List[Any],
        use_priority_IS_weight: bool = False,
        use_priority: bool = False,
        cuda: bool = False,
        device: str = 'cpu',
) -> dict:
    # data preprocess
    processes_data = {}

    action = torch.tensor(np.array([data[i]['action'] for i in range(len(data))]))
    if cuda:
        action = to_device(action, device=device)
    if action.ndim == 2 and action.shape[1] == 1:
        action = action.squeeze(1)
    processes_data['action'] = action

    obs = torch.tensor(np.array([data[i]['obs'] for i in range(len(data))]))
    if cuda:
        obs = to_device(obs, device=device)
    processes_data['obs'] = obs

    next_obs = torch.tensor(np.array([data[i]['next_obs'] for i in range(len(data))]))
    if cuda:
        next_obs = to_device(next_obs, device=device)
    processes_data['next_obs'] = next_obs

    if 'next_n_obs' in data[0]:
        next_n_obs = torch.tensor(np.array([data[i]['next_n_obs'] for i in range(len(data))]))
        if cuda:
            next_n_obs = to_device(next_n_obs, device=device)
        processes_data['next_n_obs'] = next_n_obs

    reward = torch.tensor(np.array([data[i]['reward'] for i in range(len(data))], dtype=np.float32))
    if cuda:
        reward = to_device(reward, device=device)
    reward = reward.permute(1, 0).contiguous()
    processes_data['reward'] = reward

    if 'value_gamma' in data[0]:
        value_gamma = torch.tensor(np.array([data[i]['value_gamma'] for i in range(len(data))]), dtype=torch.float32)
        if cuda:
            value_gamma = to_device(value_gamma, device=device)
        processes_data['value_gamma'] = value_gamma

    done = torch.tensor(np.array([data[i]['done'] for i in range(len(data))]), dtype=torch.float32)
    if cuda:
        done = to_device(done, device=device)
    processes_data['done'] = done

    if use_priority and use_priority_IS_weight:
        if 'priority_IS' in data:
            weight = data['priority_IS']
        else:  # for compability
            weight = data['IS']
    else:
        if 'weight' in data[0]:
            weight = torch.tensor(np.array([data[i]['weight'] for i in range(len(data))]))
        else:
            weight = None

    if weight and cuda:
        weight = to_device(weight, device=device)
    processes_data['weight'] = weight

    return processes_data


def single_env_forward_wrapper(forward_fn: Callable) -> Callable:
    """
    Overview:
        Wrap policy to support gym-style interaction between policy and single environment.
    Arguments:
        - forward_fn (:obj:`Callable`): The original forward function of policy.
    Returns:
        - wrapped_forward_fn (:obj:`Callable`): The wrapped forward function of policy.
    Examples:
        >>> env = gym.make('CartPole-v0')
        >>> policy = DQNPolicy(...)
        >>> forward_fn = single_env_forward_wrapper(policy.eval_mode.forward)
        >>> obs = env.reset()
        >>> action = forward_fn(obs)
        >>> next_obs, rew, done, info = env.step(action)

    """

    def _forward(obs):
        obs = {0: unsqueeze(to_tensor(obs))}
        action = forward_fn(obs)[0]['action']
        action = to_ndarray(squeeze(action))
        return action

    return _forward


def single_env_forward_wrapper_ttorch(forward_fn: Callable, cuda: bool = True) -> Callable:
    """
    Overview:
        Wrap policy to support gym-style interaction between policy and single environment for treetensor (ttorch) data.
    Arguments:
        - forward_fn (:obj:`Callable`): The original forward function of policy.
        - cuda (:obj:`bool`): Whether to use cuda in policy, if True, this function will move the input data to cuda.
    Returns:
        - wrapped_forward_fn (:obj:`Callable`): The wrapped forward function of policy.

    Examples:
        >>> env = gym.make('CartPole-v0')
        >>> policy = PPOFPolicy(...)
        >>> forward_fn = single_env_forward_wrapper_ttorch(policy.eval)
        >>> obs = env.reset()
        >>> action = forward_fn(obs)
        >>> next_obs, rew, done, info = env.step(action)
    """

    def _forward(obs):
        # unsqueeze means add batch dim, i.e. (O, ) -> (1, O)
        obs = ttorch.as_tensor(obs).unsqueeze(0)
        if cuda and torch.cuda.is_available():
            obs = obs.cuda()
        action = forward_fn(obs).action
        # squeeze means delete batch dim, i.e. (1, A) -> (A, )
        action = action.squeeze(0).cpu().numpy()
        return action

    return _forward
