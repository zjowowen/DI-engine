from easydict import EasyDict

from typing import Union, Optional, List, Any, Tuple, Dict
import os
import numpy as np
import torch
import torch.nn as nn

from torch.distributions import TransformedDistribution, MultivariateNormal, Independent, Categorical, Distribution
from torch.distributions.transforms import TanhTransform, SigmoidTransform

from ditk import logging

import gym

from ding.config import read_config, compile_config

from ding.utils import set_pkg_seed, SequenceType, squeeze, MODEL_REGISTRY
from ding.model.common import ReparameterizationHead, RegressionHead, DiscreteHead, MultiHead, \
    FCEncoder, ConvEncoder, OneHotEncoder


from ding.model import BaseQAC
from ding.policy import SACGeneralPolicy
from ding.envs import BaseEnvManagerV2
from ding.data import DequeBuffer
from ding.framework import task, ding_init
from ding.framework.context import OnlineRLContext
from ding.framework.middleware import data_pusher, StepCollector, interaction_evaluator, \
    CkptSaver, OffPolicyLearner, termination_checker, online_logger

from dizoo.android_env.envs.android_env import AndroidEnv

from ding.torch_utils.network.activation import build_activation

# action type of android_env:
#   TOUCH = 0
#   LIFT = 1
#   REPEAT = 2
#   KEYDOWN = 3
#   KEYUP = 4
#   KEYPRESS = 5


class multilayer_perceptron(nn.Module):

    def __init__(self, cfg):
        super(multilayer_perceptron, self).__init__()

        self.model = nn.Sequential()

        for i in range(len(cfg.hidden_sizes) - 1):
            self.model.add_module('linear' + str(i), nn.Linear(cfg.hidden_sizes[i], cfg.hidden_sizes[i + 1]))

            if isinstance(cfg.activation, list):
                self.model.add_module('activation' + str(i), build_activation(cfg.activation[i]))
            else:
                self.model.add_module('activation' + str(i), build_activation(cfg.activation))
            if hasattr(cfg, "dropout") and cfg.dropout > 0:
                self.model.add_module('dropout', nn.Dropout(cfg.dropout))
            if hasattr(cfg, "layernorm") and cfg.layernorm:
                self.model.add_module('layernorm', nn.LayerNorm(cfg.hidden_sizes[i]))

        self.model.add_module(
            'linear' + str(len(cfg.hidden_sizes) - 1), nn.Linear(cfg.hidden_sizes[-1], cfg.output_size)
        )

        if hasattr(cfg, 'final_activation'):
            self.model.add_module('final_activation', build_activation(cfg.final_activation))

        if hasattr(cfg, 'scale'):
            self.scale = nn.Parameter(torch.tensor(cfg.scale), requires_grad=False)
        else:
            self.scale = 1.0

        if hasattr(cfg, 'offset'):
            self.offset = nn.Parameter(torch.tensor(cfg.offset), requires_grad=False)
        else:
            self.offset = 0.0

        # shrink the weight of linear layer 'linear'+str(len(cfg.hidden_sizes) to it's origin 0.01
        if hasattr(cfg, 'shrink'):
            if hasattr(cfg, 'final_activation'):
                self.model[-2].weight.data.normal_(0, cfg.shrink)
                self.model[-2].bias.data.normal_(0, cfg.shrink)
            else:
                self.model[-1].weight.data.normal_(0, cfg.shrink)
                self.model[-1].bias.data.normal_(0, cfg.shrink)

    def forward(self, x):
        return self.scale * self.model(x) + self.offset

class NonegativeParameter(nn.Module):

    def __init__(self, data=None, requires_grad=True, delta=1e-8):
        super().__init__()
        if data is None:
            data = torch.zeros(1)
        self.log_data = nn.Parameter(torch.log(data + delta), requires_grad=requires_grad)

    def forward(self):
        return torch.exp(self.log_data)

    @property
    def data(self):
        return torch.exp(self.log_data)

class TanhParameter(nn.Module):

    def __init__(self, data=None, requires_grad=True):
        super().__init__()
        if data is None:
            data = torch.zeros(1)
        self.transform = TanhTransform(cache_size=1)

        self.data_inv = nn.Parameter(self.transform.inv(data), requires_grad=requires_grad)

    def forward(self):
        return self.transform(self.data_inv)

    @property
    def data(self):
        return self.transform(self.data_inv)

class NonegativeFunction(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.model = multilayer_perceptron(cfg)

    def forward(self, x):
        return torch.exp(self.model(x))

class TanhFunction(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.transform = TanhTransform(cache_size=1)
        self.model = multilayer_perceptron(cfg)

    def forward(self, x):
        return self.transform(self.model(x))

class CovarianceMatrix(nn.Module):

    def __init__(self, cfg=None, delta=1e-8):
        super().__init__()
        self.dim = cfg.dim
        self.is_diagonal_matrix = False if cfg.is_diagonal_matrix is None else cfg.is_diagonal_matrix
        self.sigma_lambda = NonegativeFunction(cfg.sigma_lambda)
        if not self.is_diagonal_matrix:
            self.sigma_offdiag = TanhFunction(cfg.sigma_offdiag)

        # register eye matrix
        self.eye = nn.Parameter(torch.eye(self.dim), requires_grad=False)
        self.delta = delta

    def low_triangle_matrix(self, x=None):
        if self.is_diagonal_matrix:
            return torch.diag_embed(self.delta + self.sigma_lambda(x))
        else:
            low_t_m = self.eye.clone()
            low_t_m = low_t_m.repeat(x.shape[0], 1, 1)
            low_t_m[torch.cat(
                (
                    torch.reshape(torch.arange(x.shape[0]).repeat(self.dim * (self.dim - 1) // 2, 1).T,
                                    (1, -1)), torch.tril_indices(self.dim, self.dim, offset=-1).repeat(1, x.shape[0])
                )
            ).tolist()] = torch.reshape(self.sigma_offdiag(x), (-1, 1)).squeeze(-1)
            low_t_m = torch.einsum(
                "bj,bjk,bk->bjk", self.delta + self.sigma_lambda(x), low_t_m, self.delta + self.sigma_lambda(x)
            )
            return low_t_m

    def forward(self, x=None):
        return torch.matmul(self.low_triangle_matrix(x), self.low_triangle_matrix(x).T)

class GaussianTanh(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.mu_model = multilayer_perceptron(cfg.mu_model)
        self.cov = CovarianceMatrix(cfg.cov)

    def dist(self, conditioning):
        mu = self.mu_model(conditioning)
        # repeat the sigma to match the shape of mu
        scale_tril = self.cov.low_triangle_matrix(conditioning)
        return TransformedDistribution(
            base_distribution=MultivariateNormal(loc=mu, scale_tril=scale_tril),
            transforms=[TanhTransform(cache_size=1)]
        )

    def log_prob(self, x, conditioning):
        return self.dist(conditioning).log_prob(x)

    def sample(self, conditioning, sample_shape=torch.Size()):
        return self.dist(conditioning).sample(sample_shape=sample_shape)

    def rsample(self, conditioning, sample_shape=torch.Size()):
        return self.dist(conditioning).rsample(sample_shape=sample_shape)

    def rsample_and_log_prob(self, conditioning, sample_shape=torch.Size()):
        dist = self.dist(conditioning)
        x = dist.rsample(sample_shape=sample_shape)
        log_prob = dist.log_prob(x)
        return x, log_prob

    def sample_and_log_prob(self, conditioning, sample_shape=torch.Size()):
        with torch.no_grad():
            return self.rsample_and_log_prob(conditioning, sample_shape)

    def entropy(self, conditioning):
        mu = self.mu_model(conditioning)
        # repeat the sigma to match the shape of mu
        scale_tril = self.cov.low_triangle_matrix(conditioning)
        base_distribution = MultivariateNormal(loc=mu, scale_tril=scale_tril)
        x = base_distribution.rsample(sample_shape=torch.Size([1000]))
        return base_distribution.entropy() + torch.sum(torch.log(1.0 - torch.tanh(x) ** 2), dim=(0, 2)) / 1000

    def forward(self, conditioning):
        dist = self.dist(conditioning)
        x = dist.rsample()
        log_prob = dist.log_prob(x)
        return x, log_prob

class GaussianSigmoid(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.mu_model = multilayer_perceptron(cfg.mu_model)
        self.cov = CovarianceMatrix(cfg.cov)

    def dist(self, conditioning):
        mu = self.mu_model(conditioning)
        # repeat the sigma to match the shape of mu
        scale_tril = self.cov.low_triangle_matrix(conditioning)
        return TransformedDistribution(
            base_distribution=MultivariateNormal(loc=mu, scale_tril=scale_tril),
            transforms=[SigmoidTransform(cache_size=1)]
        )

    def log_prob(self, x, conditioning):
        return self.dist(conditioning).log_prob(x)

    def sample(self, conditioning, sample_shape=torch.Size()):
        return self.dist(conditioning).sample(sample_shape=sample_shape)

    def rsample(self, conditioning, sample_shape=torch.Size()):
        return self.dist(conditioning).rsample(sample_shape=sample_shape)

    def rsample_and_log_prob(self, conditioning, sample_shape=torch.Size()):
        dist = self.dist(conditioning)
        x = dist.rsample(sample_shape=sample_shape)
        log_prob = dist.log_prob(x)
        return x, log_prob

    def sample_and_log_prob(self, conditioning, sample_shape=torch.Size()):
        with torch.no_grad():
            return self.rsample_and_log_prob(conditioning, sample_shape)

    def entropy(self, conditioning):
        mu = self.mu_model(conditioning)
        # repeat the sigma to match the shape of mu
        scale_tril = self.cov.low_triangle_matrix(conditioning)
        base_distribution = MultivariateNormal(loc=mu, scale_tril=scale_tril)
        x = base_distribution.rsample(sample_shape=torch.Size([1000]))
        return base_distribution.entropy() + torch.sum(torch.log(1.0 - torch.tanh(x) ** 2), dim=(0, 2)) / 1000

    def forward(self, conditioning):
        dist = self.dist(conditioning)
        x = dist.rsample()
        log_prob = dist.log_prob(x)
        return x, log_prob

class CategoricalModule(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = multilayer_perceptron(cfg.mlp)

    def sample(self, conditioning):
        dist = Categorical(logits=self.model(conditioning))
        return dist.sample()
    
    def log_prob(self, x, conditioning):
        dist = Categorical(logits=self.model(conditioning))
        return dist.log_prob(x)
    
    def entropy(self, conditioning):
        dist = Categorical(logits=self.model(conditioning))
        return dist.entropy()

    def forward(self, condition):
        dist = Categorical(logits=self.model(condition))
        x = dist.sample()
        log_prob = dist.log_prob(x)
        return x, log_prob


apple_flinger_sac_config = dict(
    exp_name='apple_flinger_sac',
    env=dict(
        collector_env_num=1,
        evaluator_env_num=1,
        env_id='apple_flinger',
        n_evaluator_episode=1,
        stop_value=100000,
        avd_name="my_avd",
        android_avd_home="/home/zjow/.android/avd",
        android_sdk_root="/home/zjow/Android/Sdk",
        emulator_path="/home/zjow/Android/Sdk/emulator/emulator",
        adb_path="/home/zjow/Android/Sdk/platform-tools/adb",
        task_path="/home/zjow/software/android_game/apple_flinger/apple_flinger_M_1_1.textproto",
        run_headless=False,
    ),
    policy=dict(
        cuda=True,
        action_space='general',
        model=dict(
            action_space='general',
            actor=dict(
                encoder_output_size=420,
                activation='relu',
                action_type_net=dict(
                    action_type_num=3,
                    mlp=dict(
                        hidden_sizes=[420, 64],
                        output_size=3,
                        activation='relu',
                        shrink=0.01,
                    ),
                ),
                gaussian_net=dict(
                    mu_model=dict(
                        hidden_sizes=[420, 64, 16],
                        output_size=2,
                        activation='relu',
                    ),
                    cov=dict(
                        dim=2,
                        is_diagonal_matrix=True,
                        sigma_lambda=dict(
                            hidden_sizes=[420, 64, 16],
                            output_size=2,
                            activation='relu',
                        ),
                    ),
                ),
            ),
            critic=dict(
                encoder_output_size=420,
                activation='relu',
                model=dict(
                    hidden_sizes=[420, 128, 64],
                    output_size=1,
                    activation='relu',
                ),
            ),
        ),
        learn=dict(
            target_entropy=-2,
            batch_size=128,
        ),
        random_collect_size=200,
        collect=dict(
            n_sample=5,
            discount_factor=0.99,
            gae_lambda=0.95,
            collector=dict(collect_print_freq=1000, ),
        ),
        eval=dict(evaluator=dict(eval_freq=200, ), ),
        other=dict(replay_buffer=dict(replay_buffer_size=500, ), ),
    ),
)
apple_flinger_sac_config = EasyDict(apple_flinger_sac_config)
main_config = apple_flinger_sac_config

apple_flinger_sac_create_config = dict(
    env=dict(
        type='android_env',
        import_names=['dizoo.android_env.envs.android_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='sac_general'),
)
apple_flinger_sac_create_config = EasyDict(apple_flinger_sac_create_config)
create_config = apple_flinger_sac_create_config

class GaussianFourierProjectionTimeEncoder(nn.Module):
    """
    Overview:
        Gaussian random features for encoding time steps.
        This module is used as the encoder of time in generative models such as diffusion model.
    Interfaces:
        ``__init__``, ``forward``.
    """

    def __init__(self, embed_dim, scale=30.):
        """
        Overview:
            Initialize the Gaussian Fourier Projection Time Encoder according to arguments.
        Arguments:
            - embed_dim (:obj:`int`): The dimension of the output embedding vector.
            - scale (:obj:`float`): The scale of the Gaussian random features.
        """
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale * 2 * np.pi, requires_grad=False)

    def forward(self, x):
        """
        Overview:
            Return the output embedding vector of the input time step.
        Arguments:
            - x (:obj:`torch.Tensor`): Input time step tensor.
        Returns:
            - output (:obj:`torch.Tensor`): Output embedding vector.
        Shapes:
            - x (:obj:`torch.Tensor`): :math:`(B,)`, where B is batch size.
            - output (:obj:`torch.Tensor`): :math:`(B, embed_dim)`, where B is batch size, embed_dim is the \
                dimension of the output embedding vector.
        Examples:
            >>> encoder = GaussianFourierProjectionTimeEncoder(128)
            >>> x = torch.randn(100)
            >>> output = encoder(x)
        """
        x_proj = x[..., None] * self.W[None, :]
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class AndroidEnvEncoder(nn.Module):

    def __init__(self, output_size, activation, input_spec: gym.spaces.Dict):
        super(AndroidEnvEncoder, self).__init__()
        self.model_dict = nn.ModuleDict()
        
        output_size = output_size // len(input_spec.keys())
        self.real_output_size = 0

        for k, v in input_spec.items():
            if isinstance(v, gym.spaces.Box):
                if len(v.shape) == 1:
                    # add a MultiLayerPerceptron
                    if v.dtype == np.float32:
                        self.model_dict[k] = FCEncoder(
                            obs_shape=v.shape[0],
                            hidden_size_list=[output_size, output_size],
                            activation=nn.SiLU(),
                        )
                        self.real_output_size += output_size
                    elif v.dtype == np.uint8:
                        self.model_dict[k] = nn.Identity()
                        self.real_output_size += v.shape[0]
                elif len(v.shape) == 3:
                    # add a ConvolutionalNeuralNetwork
                    image_shape=list(v.shape)
                    # swap the channel axis to the first
                    image_shape[0], image_shape[2] = image_shape[2], image_shape[0]
                    conv_encoder=ConvEncoder(
                        obs_shape=image_shape,
                        channel_first=False,
                    )
                    self.model_dict[k] = conv_encoder
                    self.real_output_size += conv_encoder.output_size
                elif len(v.shape) == 0:
                    if isinstance(v, gym.spaces.Discrete):
                        self.model_dict[k] = OneHotEncoder(obs_shape=v.n)
                        self.real_output_size += v.n
                    elif isinstance(v, gym.spaces.Box):
                        embbed_dim = output_size
                        self.model_dict[k] = GaussianFourierProjectionTimeEncoder(
                            embed_dim=embbed_dim,
                            scale=0.01,
                        )
                        self.real_output_size += embbed_dim
                else:
                    raise NotImplementedError
            elif isinstance(v, gym.spaces.Discrete):
                self.model_dict[k] = OneHotEncoder(obs_shape=v.n)
                self.real_output_size += v.n
            else:
                raise NotImplementedError
            
    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Overview:
            Forward computation graph of the encoder.
        Arguments:
            - x (:obj:`dict[str, torch.Tensor]`): The input dict of observation tensor data.
        Returns:
            - output (:obj:`torch.Tensor`): The encoded tensor data.
        Shapes:
            - x (:obj:`dict[str, torch.Tensor]`): :math:`(B, N)`, where B is batch size and N is the dimension of \
                observation space.
            - output (:obj:`torch.Tensor`): :math:`(B, N)`, where B is batch size and N is the dimension of \
                encoded space.
        """
        output = []
        for i, (k, v) in enumerate(x.items()):
            output.append(self.model_dict[k](v))
        output = torch.cat(output, dim=-1)
        return output


class AndroidEnvActor(nn.Module):

    def __init__(self, cfg, observation_spec: gym.spaces.Dict, action_spec: gym.spaces.Dict):
        super(AndroidEnvActor, self).__init__()
        self.cfg = cfg
        self.observation_spec = observation_spec
        self.action_spec = action_spec
        self.encoder = AndroidEnvEncoder(cfg.encoder_output_size, activation=cfg.activation, input_spec=observation_spec)
        cfg.action_type_net.mlp.hidden_sizes[0]=self.encoder.real_output_size
        cfg.gaussian_net.mu_model.hidden_sizes[0]=self.encoder.real_output_size
        cfg.gaussian_net.cov.sigma_lambda.hidden_sizes[0]=self.encoder.real_output_size

        self.action_type_net = CategoricalModule(cfg.action_type_net)
        self.touch_position_net = GaussianSigmoid(cfg.gaussian_net)

    def forward(self, x: dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Overview:
            Forward computation graph of the actor.
        Arguments:
            - x (:obj:`dict[str, torch.Tensor]`): The input dict of observation tensor data.
        Returns:
            - action (:obj:`torch.Tensor`): The action tensor data.
            - log_prob (:obj:`torch.Tensor`): The log probability of the action.
            - entropy (:obj:`torch.Tensor`): The entropy of the action.
        Shapes:
            - x (:obj:`dict[str, torch.Tensor]`): :math:`(B, N)`, where B is batch size and N is the dimension of \
                observation space.
            - action (:obj:`torch.Tensor`): :math:`(B, N)`, where B is batch size and N is the dimension of \
                action space.
            - log_prob (:obj:`torch.Tensor`): :math:`(B, )`, where B is batch size.
            - entropy (:obj:`torch.Tensor`): :math:`(B, )`, where B is batch size.
        """
        x = self.encoder(x)
        action_type, log_prob_type = self.action_type_net(x)
        touch_position, log_prob_position = self.touch_position_net(x)
        action = {'action_type': action_type, 'touch_position': touch_position}
        log_prob = log_prob_type + log_prob_position
        return action, log_prob
    
class AndroidEnvCritic(nn.Module):

    def __init__(self, cfg, observation_spec: gym.spaces.Dict, action_spec: gym.spaces.Dict):
        super(AndroidEnvCritic, self).__init__()
        self.cfg = cfg
        self.observation_spec = observation_spec
        self.action_spec = action_spec
        self.obs_encoder = AndroidEnvEncoder(cfg.encoder_output_size, activation=cfg.activation, input_spec=observation_spec)
        self.action_encoder = AndroidEnvEncoder(cfg.encoder_output_size, activation=cfg.activation, input_spec=action_spec)
        cfg.model.hidden_sizes[0]=self.obs_encoder.real_output_size + self.action_encoder.real_output_size
        self.model = multilayer_perceptron(cfg.model)
        if hasattr(cfg, 'double_q') and not cfg.double_q:
            self.double_q = False
        else:
            self.double_q = True
            self.obs_encoder_2 = AndroidEnvEncoder(cfg.encoder_output_size, activation=cfg.activation, input_spec=observation_spec)
            self.action_encoder_2 = AndroidEnvEncoder(cfg.encoder_output_size, activation=cfg.activation, input_spec=action_spec)
            self.model_2 = multilayer_perceptron(cfg.model)


    def min_q(self, obs: dict[str, torch.Tensor], action: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Overview:
            Compute the minimum Q value of the two Q value networks.
        Arguments:
            - obs (:obj:`dict[str, torch.Tensor]`): The input dict of observation tensor data.
            - action (:obj:`dict[str, torch.Tensor]`): The input dict of action tensor data.
        Returns:
            - min_q (:obj:`torch.Tensor`): The minimum Q value tensor data.
        Shapes:
            - obs (:obj:`dict[str, torch.Tensor]`): :math:`(B, N)`, where B is batch size and N is the dimension of \
                observation space.
            - action (:obj:`dict[str, torch.Tensor]`): :math:`(B, N)`, where B is batch size and N is the dimension of \
                action space.
            - min_q (:obj:`torch.Tensor`): :math:`(B, )`, where B is batch size.
        """
        q_value = self.forward(obs, action)
        if self.double_q:
            q_value = torch.min(q_value,1).values
        return q_value


    def forward(self, obs: dict[str, torch.Tensor], action: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Overview:
            Forward computation graph of the critic.
        Arguments:
            - obs (:obj:`dict[str, torch.Tensor]`): The input dict of observation tensor data.
            - action (:obj:`dict[str, torch.Tensor]`): The input dict of action tensor data.
        Returns:
            - q_value (:obj:`torch.Tensor`): The Q value tensor data.
        Shapes:
            - obs (:obj:`dict[str, torch.Tensor]`): :math:`(B, N)`, where B is batch size and N is the dimension of \
                observation space.
            - action (:obj:`dict[str, torch.Tensor]`): :math:`(B, N)`, where B is batch size and N is the dimension of \
                action space.
            - q_value (:obj:`torch.Tensor`): :math:`(B, )`, where B is batch size.
        """
        obs_emb = self.obs_encoder(obs)
        action_emb = self.action_encoder(action)
        q_value = self.model(torch.cat([obs_emb, action_emb], dim=-1))
        if self.double_q:
            obs_emb_2 = self.obs_encoder_2(obs)
            action_emb_2 = self.action_encoder_2(action)
            q_value_2 = self.model_2(torch.cat([obs_emb_2, action_emb_2], dim=-1))
            q_value = torch.cat([q_value, q_value_2], dim=-1)
        return q_value


def main():
    logging.getLogger().setLevel(logging.INFO)
    cfg = compile_config(main_config, create_cfg=create_config, auto=True)
    ding_init(cfg)
    with task.start(async_mode=False, ctx=OnlineRLContext()):
        collector_env = BaseEnvManagerV2(
            env_fn=[lambda: AndroidEnv(cfg.env) for _ in range(cfg.env.collector_env_num)], cfg=cfg.env.manager
        )
        # evaluator_env = BaseEnvManagerV2(
        #     env_fn=[lambda: AndroidEnv(cfg.env) for _ in range(cfg.env.evaluator_env_num)], cfg=cfg.env.manager
        # )

        set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

        actor_model=AndroidEnvActor(cfg.policy.model.actor, collector_env.observation_space, collector_env.action_space)
        critic_model=AndroidEnvCritic(cfg.policy.model.critic, collector_env.observation_space, collector_env.action_space)
        model = BaseQAC(actor=actor_model, critic=critic_model)
        buffer_ = DequeBuffer(size=cfg.policy.other.replay_buffer.replay_buffer_size)
        policy = SACGeneralPolicy(cfg.policy, model=model)

        # task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
        task.use(
            StepCollector(cfg, policy.collect_mode, collector_env, random_collect_size=cfg.policy.random_collect_size)
        )
        task.use(data_pusher(cfg, buffer_))
        task.use(OffPolicyLearner(cfg, policy.learn_mode, buffer_))
        task.use(CkptSaver(policy, cfg.exp_name, train_freq=100))
        task.use(termination_checker(max_train_iter=10000))
        task.use(online_logger())
        task.run()

if __name__ == "__main__":
    main()
