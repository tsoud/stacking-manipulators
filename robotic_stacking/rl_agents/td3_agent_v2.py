from typing import List, Optional, Union

import tensorflow as tf
from tf_agents.agents import ddpg, td3
from tf_agents.environments import py_environment, tf_py_environment
from tf_agents.train.utils import spec_utils, strategy_utils, train_utils
from tf_agents.typing import types
from tf_agents.utils import common

from robotic_stacking.training.training_utils import (
    IncorrectEnvironmentType
)

# ----------------------------------------------------------------------------


class tfa_td3_agent:
    """
    Configure and create a TF-Agents TD3 agent.

    Configures NN parameters for actor and critic and agent-specific 
    parameters.

    keyword args:
    ------------
    tfa_env: the stacking environment to use. The TD3 agent can use 
        either `py_environment` or `tf_py_environment` versions of 
        the stacking environment.
    actor_net_params, critic_net_params: Layer parameters of actor and 
        critic networks passed through to network constructors with 
        the following `dict` structure (use `None` for defaults):
        {'preprocessing_layers': `conv` or `fc` layers,
        'conv_layer_params': [(filters, kernel, stride), ...], 
        'fc_layer_params': [n_units_layer1, ...], 
        'dropout_layer_params': [fraction_fc_layer1, ...]}
        for critic network observation, action and joint layers can be 
        defined:
        {
            'observation_conv_layer_params': [
                (filters, kernel, stride), ...
                ], 
            'observation_fc_layer_params': [n_units_layer1, ...], 
                ...
            'action_fc_layer_params: [n_units_layer1, ...],
                ...
            'joint_fc_layer_params': [fraction_fc_layer1, ...], 
                ...
        }
    actor_activation_fn: Activation function for actor network layers.
    critic_activation_fn: If `None`, use same activation function as 
        actor otherwise use a specific function for the critic network 
        layers.
    critic_output_activation: Last layer activation function (if used).
    actor_kernel_initializer: Initializer for actor conv and fc layers.
    actor_last_kernel_init: 
    critic_kernel_initializer: Initializer for critic conv and 
        fc layers. Default (`None`) is same as actor initializer.
    critic_last_kernel_init: Initializer for value regression layer. 
        Default (`None`) is `RandomUniform`.
    use_gpu, use_tpu: sets up a `tf.distribute.Strategy` using 
        TF-agents `strategy_utils` for multiples GPUs or TPUs.
        If none (or only a single GPU) are available, these args are
        forced to `False`.
    """
    def __init__(self,
                 tfa_env:py_environment.PyEnvironment,
                 actor_net_params:dict,
                 critic_net_params:dict,
                 actor_activation_fn=tf.keras.activations.relu,
                 critic_activation_fn:Optional=None,
                 critic_output_activation:Optional=None,
                 actor_kernel_initializer='variance_scaling',
                 actor_last_kernel_init='random_uniform',
                 critic_kernel_initializer:Optional=None,
                 critic_last_kernel_init:Optional=None,
                 use_gpu=False, 
                 use_tpu=False):

        self.collect_env = tfa_env
        self._observation_spec, self._action_spec, self._time_step_spec = (
            spec_utils.get_tensor_specs(self.collect_env)
        )
        self.eval_env = tfa_env.copy_env()
        self.actor_net_params = actor_net_params
        self.critic_net_params = critic_net_params
        self.actor_activation = actor_activation_fn
        self.critic_activation = critic_activation_fn or actor_activation_fn
        self.critic_output_activation = critic_output_activation
        self.actor_kernel_initializer = actor_kernel_initializer
        self.actor_last_kernel_init = actor_last_kernel_init
        self.critic_kernel_initializer = (
            critic_kernel_initializer or self.actor_kernel_initializer
        )
        self.critic_last_kernel_init = (
            critic_last_kernel_init or self.critic_kernel_initializer
        )
        # check if there are multiple GPUs
        gpu_devices = tf.config.list_physical_devices('GPU')
        self._use_gpu = False if len(gpu_devices) <= 1 else use_gpu
        tpu_devices = tf.config.list_physical_devices('TPU')
        self._use_tpu = False if len(tpu_devices) < 1 else use_tpu
        self.strategy = strategy_utils.get_strategy(
            tpu=self._use_gpu, use_gpu=self._use_tpu)
        # the following attributes are defined when relevant methods are called
        self._train_step = None
        self._actor_net = None
        self._critic_net = None
        self._agent = None

    def make_actor_net(self, overwrite:bool=False):
        """
        Create the actor network for training the agent.
        
        overwrite: whether to overwrite the existing network if one was
            previously defined.
        """
        if (self._actor_net is None) or overwrite:
            with self.strategy.scope():
                self._actor_net = ddpg.actor_network.ActorNetwork(
                    input_tensor_spec=self._observation_spec,
                    output_tensor_spec=self._action_spec,
                    activation_fn=self.actor_activation,
                    kernel_initializer=self.actor_kernel_initializer, 
                    last_kernel_initializer=self.actor_last_kernel_init,
                    **self.actor_net_params
                )
        else:
            return (
                "An actor network is already defined. Set `overwrite` to" 
                + " `True` if you would like to overwrite it."
            )

    @property
    def actor_net(self):
        if self._actor_net is None:
            return (
                'No actor network defined. Run `make_actor_net()` '
                + 'to define network.'
            )
        return self._actor_net

    def make_critic_net(self, overwrite=False):
        """
        Create the agent's critic network.
        
        overwrite: whether to overwrite the existing network if one was
            previously defined.
        """
        if (self._critic_net is None) or overwrite:
            with self.strategy.scope():
                self._critic_net = ddpg.critic_network.CriticNetwork(
                    input_tensor_spec=(self._observation_spec,
                                       self._action_spec),
                    activation_fn=self.critic_activation,
                    output_activation_fn=self.critic_output_activation,
                    kernel_initializer=self.critic_kernel_initializer,
                    last_kernel_initializer=self.critic_last_kernel_init,
                    **self.critic_net_params
                )
        else:
            return (
                "A critic network is already defined. Set `overwrite` to " 
                + " `True` if you would like to overwrite it."
            )

    @property
    def critic_net(self):
        if self._critic_net is None:
            return (
                'No critic network defined. Run `make_critic_net()` ' 
                + 'to define network.'
            )
        return self._critic_net

    def make_actor_critic_nets(self, overwrite_all=False):
        """
        Convenience function to create both networks at the same time.
        
        overwrite_all: whether to overwrite all existing networks if 
            any were previously defined. If set to `False`, only a 
            network that was not already created is defined.
        """
        if (self._actor_net is None) or overwrite_all:
            self.make_actor_net()
        if (self._critic_net is None) or overwrite_all:
            self.make_critic_net()     
        else:
            return (
                "Networks were already defined. Set `overwrite_all` to `True`"
                + " if you would like to overwrite existing networks."
            )

    def make_agent(self, 
                   actor_optimizer:types.Optimizer,
                   critic_optimizer:Optional[types.Optimizer]=None,
                   exploration_stddev=0.1,
                   td_errors_loss_fn:Optional[types.LossFn]=None,
                   target_update_tau:types.Float=0.005,
                   target_update_period:types.Int=1,
                   actor_update_period:types.Int=1,
                   reward_scaling:types.Float=1.0,
                   gamma:types.Float=0.99,
                   target_policy_noise:types.Float=0.2,
                   target_policy_noise_clip:types.Float=0.5,
                   name:Optional[str]=None,
                   **kwargs):
        """
        Define agent parameters and create the TD3 agent.

        Details of the TD3 algorithm can be found in the original paper
        by Fujimoto et al. "Addressing Function Approximation Error in 
        Actor-Critic Methods".

        keyword args:
        ------------
        actor_optimizer: Optimizer (e.g. `tf.keras.optimizers.Adam()`) 
            for actor network gradient updates.
        critic_optimizer: Optimizer for critic network. 
            If `None`, use `actor_optimizer`.
        alpha_optimizer: Optimizer for alpha (temperature parameter 
            for entropy regularization). Uses `actor_optimizer` 
            when `None`.
        exploration_stddev: standard deviation of noise added to 
            exploration policy actions. Default is 0.1 per original 
            paper.
        td_errors_loss_fn: Function for calculating TD loss. 
            Default is element-wise Huber loss.
        target_update_tau: Smoothing coefficient for soft (Polyak) 
            target updates.
        target_update_period: Interval for soft updates.
        actor_update_period: Interval for performing actor network 
            optimization.
        reward_scaling: Multiplicative scale factor for rewards. 
            Typically not required with auto temperature adjustment.
        gamma: Discount factor.
        target_policy_noise: noise added to target policy actions. 
            Default is N(0., 0.2) per original paper.
        target_policy_noise_clip: clip the target policy to +/- this 
            value. Default is +/- 0.5 per original paper.
        name: Optional name for agent. Default is class name.
        kwargs: additional kwargs passed to `tf_agents.agents.Td3Agent`
        """
        if self._agent is not None:
            new = input('Agent is already defined,'\
                ' create a new agent (Y/N)?')
            if new[0] in 'Nn':
                new = False
                return 'Skipped agent creation.'
            elif new[0] in 'Yy':
                new = True
            else:
                raise ValueError(f'{new} is not a valid Y/N response.')

        if self._actor_net is None:
            self.make_actor_net()
        if self._critic_net is None:
            self.make_critic_net()

        critic_optimizer = critic_optimizer or actor_optimizer

        if (self._agent is None) or new:
            with self.strategy.scope():
                # Create the train step counter
                self._train_step = train_utils.create_train_step()
                self._agent = td3.td3_agent.Td3Agent(
                    time_step_spec=self._time_step_spec,
                    action_spec=self._action_spec, 
                    actor_network=self._actor_net,
                    critic_network=self._critic_net, 
                    actor_optimizer=actor_optimizer,
                    critic_optimizer=critic_optimizer,
                    exploration_noise_std=exploration_stddev,
                    td_errors_loss_fn=td_errors_loss_fn,
                    target_update_tau=target_update_tau,
                    target_update_period=target_update_period,
                    actor_update_period=actor_update_period,
                    reward_scale_factor=reward_scaling,
                    gamma=gamma,
                    target_policy_noise=target_policy_noise,
                    target_policy_noise_clip=target_policy_noise_clip,
                    train_step_counter=self._train_step,
                    name=name,
                    **kwargs
                )
                self._agent.initialize()

    @property
    def agent(self):
        if self._agent is None:
            return 'No agent defined. Use `make_agent()` to create one.'
        return self._agent

    @property
    def collect_data_spec(self):
        """
        Get tensor spec for replay buffer.
        """
        if self._agent is None:
            return 'No agent defined. Use `make_agent()` to create one.'
        return self._agent.collect_data_spec