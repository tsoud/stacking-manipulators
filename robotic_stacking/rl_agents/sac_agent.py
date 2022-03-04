from typing import List, Optional

import tensorflow as tf
from tf_agents.agents import ddpg, sac
from tf_agents.environments import py_environment, tf_py_environment
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.networks import actor_distribution_network
from tf_agents.train.utils import spec_utils, strategy_utils, train_utils
from tf_agents.typing import types

# --------------------------------------------------------------------------- #


class tfa_sac_agent:
    """
    Define a TF-Agents Soft Actor-Critic agent.

    Configures NN parameters for actor and critic, distribution 
    strategy (CPU/GPU/TPU), and agent-specific parameters.

    keyword args:
    ------------
    tfa_env: the stacking environment to use.
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
    critic_kernel_initializer: Initializer for critic conv and 
        fc layers. Default (`None`) is same as actor initializer.
    critic_last_kernel_init: Initializer for value regression layer. 
        Default (`None`) is `RandomUniform`.
    actor_dtype: TF data type for actor convolutional and fc layers.
    use_gpu: Applies GPU strategy with available GPUs. 
        If `False`, use CPU.
    use_tpu: Applies TPU strategy if available.
    """
    def __init__(self,
                 tfa_env:py_environment.PyEnvironment,
                 actor_net_params:dict,
                 critic_net_params:dict,
                 actor_activation_fn=tf.keras.activations.relu,
                 critic_activation_fn:Optional=None,
                 critic_output_activation:Optional=None,
                 actor_kernel_initializer='glorot_uniform',
                 critic_kernel_initializer:Optional=None,
                 critic_last_kernel_init:Optional=None,
                 actor_dtype=tf.float32,
                 use_gpu=True, 
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
        self.critic_kernel_initializer = (
            critic_kernel_initializer or self.actor_kernel_initializer
        )
        self.critic_last_kernel_init = (
            critic_last_kernel_init or self.critic_kernel_initializer
        )
        self.actor_layers_dtype = actor_dtype
        self.strategy = strategy_utils.get_strategy(
            tpu=use_tpu, use_gpu=use_gpu)
        # the following attributes are defined when relevant methods are called
        self._train_step = None
        self._actor_net = None
        self._critic_net = None
        self._agent = None

    def make_actor_net(self):
        """Create the actor network for training the agent."""
        if self._actor_net:
            overwrite = input('Overwrite existing actor network (Y/N)?')
            if overwrite[0] in 'Nn':
                overwrite = False
                return 'Skipped network creation.'
            elif overwrite[0] in 'Yy':
                overwrite = True
            else:
                raise ValueError(f'{overwrite} is not a valid Y/N response.')

        if (self._actor_net is None) or overwrite:
            with self.strategy.scope():
                self._actor_net = (
                    actor_distribution_network.ActorDistributionNetwork(
                        input_tensor_spec=self._observation_spec,
                        output_tensor_spec=self._action_spec,
                        continuous_projection_net=(
                            tanh_normal_projection_network
                            .TanhNormalProjectionNetwork
                        ),
                        activation_fn=self.actor_activation,
                        kernel_initializer=self.actor_kernel_initializer,
                        dtype=self.actor_layers_dtype,
                        **self.actor_net_params
                    )
                )

    @property
    def actor_net(self):
        if self._actor_net is None:
            return (
                'No actor network defined. Run `make_actor_net()` '
                + 'to define network.'
            )
        return self._actor_net

    def make_critic_net(self):
        """Create the agent's critic network."""
        if self._critic_net:
            overwrite = input('Overwrite existing critic network (Y/N)?')
            if overwrite[0] in 'Nn':
                overwrite = False
                return 'Skipped network creation.'
            elif overwrite[0] in 'Yy':
                overwrite = True
            else:
                raise ValueError(f'{overwrite} is not a valid Y/N response.')

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

    @property
    def critic_net(self):
        if self._critic_net is None:
            return (
                'No critic network defined. Run `make_critic_net()` ' 
                + 'to define network.'
            )
        return self._critic_net

    def make_actor_critic_nets(self):
        """Convenience function to create both networks at the same time."""
        if self._actor_net or self._critic_net:
            overwrite = input('Overwrite existing networks (Y/N)?')
            if overwrite[0] in 'Nn':
                return 'Skipped network creation.'
            elif overwrite[0] in 'Yy':
                self._actor_net, self._critic_net = None, None
            else:
                raise ValueError(f'{overwrite} is not a valid Y/N response.')

        self.make_actor_net()
        self.make_critic_net()

    def make_agent(self, 
                   actor_optimizer:types.Optimizer, 
                   critic_optimizer:Optional[types.Optimizer]=None, 
                   alpha_optimizer:Optional[types.Optimizer]=None, 
                   td_errors_loss_fn:types.LossFn=tf.math.squared_difference, 
                   loss_weights:List[types.Float]=[1.0, 0.5, 1.0], 
                   target_update_tau:types.Float=0.005, 
                   target_update_period:types.Int=1, 
                   target_entropy:Optional[types.Float]=None,  
                   reward_scaling:types.Float=1.0, 
                   gamma:types.Float=0.99, 
                   use_log_alpha:bool=True, 
                   name:Optional[str]=None, 
                   **kwargs):
        """
        Create the SAC agent.

        Details of the SAC algorithm can be found in the original paper
        by Haarnoja et al. "Soft Actor-Critic Algorithms and 
        Applications".

        keyword args:
        ------------
        actor_optimizer: Optimizer (e.g. `tf.keras.optimizers.Adam()`) 
            for actor network gradient updates.
        critic_optimizer: Optimizer for critic network. 
            If `None`, use `actor_optimizer`.
        alpha_optimizer: Optimizer for alpha (temperature parameter 
            for entropy regularization). Uses `actor_optimizer` 
            when `None`.
        td_errors_loss_fn: Function for calculating TD loss.
        loss_weights: List of weights to apply to loss values given as 
            [`actor_loss_wt`, `critic_loss_wt`, `alpha_loss_wt`]
        target_update_tau: Smoothing coefficient for soft (Polyak) 
            target updates.
        target_update_period: Interval for soft updates.
        target_entropy: Entropy target for automatic temperature 
            adjustment. Default `None` uses -1*dim(actions) as in 
            Haarnoja et al. reference implementation.
        reward_scaling: Multiplicative scale factor for rewards. 
            Typically not required with auto temperature adjustment.
        gamma: Discount factor.
        use_log_alpha: Use log_alpha instead of alpha to calculate 
            alpha loss.
        name: Optional name for agent. Default is class name.
        kwargs: additional kwargs passed to `tf_agents.agents.SacAgent`
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
        alpha_optimizer = alpha_optimizer or actor_optimizer
        actor_loss_wt, critic_loss_wt, alpha_loss_wt = loss_weights

        if (self._agent is None) or new:
            with self.strategy.scope():
                # Create the train step counter
                self._train_step = train_utils.create_train_step()
                self._agent = sac.sac_agent.SacAgent(
                    time_step_spec=self._time_step_spec, 
                    action_spec=self._action_spec, 
                    actor_network=self._actor_net, 
                    critic_network=self._critic_net, 
                    actor_optimizer=actor_optimizer, 
                    critic_optimizer=critic_optimizer, 
                    alpha_optimizer=alpha_optimizer, 
                    actor_loss_weight=actor_loss_wt, 
                    critic_loss_weight=critic_loss_wt, 
                    alpha_loss_weight=alpha_loss_wt, 
                    target_update_tau=target_update_tau, 
                    target_update_period=target_update_period,
                    td_errors_loss_fn=td_errors_loss_fn,
                    target_entropy=target_entropy, 
                    reward_scale_factor=reward_scaling, 
                    gamma=gamma, 
                    use_log_alpha_in_alpha_loss=use_log_alpha, 
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
