import os
import pprint
import tempfile
import time
from datetime import timedelta
from functools import wraps
from typing import Callable, List, Optional, Union

import reverb
from reverb import rate_limiters
import tensorflow as tf
import tqdm
from tf_agents import metrics, policies, replay_buffers
from tf_agents.drivers import dynamic_step_driver, py_driver
from tf_agents.eval import metric_utils
from tf_agents.replay_buffers import reverb_replay_buffer, reverb_utils
from tf_agents.train import actor, learner, triggers
from tf_agents.train.utils import spec_utils, strategy_utils
from tf_agents.train.utils import train_utils as tfa_train_utils
from tf_agents.typing import types as tfa_types
from tf_agents.utils import common


from robotic_stacking.replay_buffers import reverb_buffer
from robotic_stacking.rl_agents import td3_agent
from robotic_stacking.training.training_utils import (
    NoReplayBufferException, 
    MissingDriverException,
    measure_run_time
)

# ----------------------------------------------------------------------------


class td3_trainer:
    """
    Configure and run training with a TD3 model.

    keyword args:
    ------------
    agent_class: Instance of `tfa_sac_agent` containing the agent 
        to train.
    replay_buffer: An instance of `reverb_buffer` or any buffer 
        from `tf_agents.replay_buffers.replay_buffer.ReplayBuffer`
    initial_replay_steps: Number of steps for collecting initial 
        samples for the replay memory before training begins.
    num_eval_episodes: Number of episodes to aggregate for evaluation.
    train_steps_per_iteration: Number of times to train the networks 
        with each training iteration.
    collection_steps_per_run: Number of environment steps to iterate 
        over when evaluating the main policy actor.
    add_eval_metrics: Additional evaluation metrics. Default metrics 
        are average episode length and average reward.When specifying 
        additional metrics, it is important to make sure they are 
        compatible with the environment type.
    save_dir: Directory to save the model. If `None`, the model is 
        saved to a `tempdir` object.
    save_interval: How often to save the policy.
    use_tf_functions: wrap functions into a TF graph when possible, 
        equivalent to using `@tf.function` in standard TensorFlow.
    """
    def __init__(self,  
                 agent:td3_agent.tfa_td3_agent,
                 replay_buffer:Union[
                     reverb_buffer.reverb_buffer, 
                     replay_buffers.replay_buffer.ReplayBuffer
                     ],
                 initial_replay_steps:int,
                 train_steps_per_iteration:int=1,
                 collection_steps_per_run:int=1,
                 add_eval_metrics:list=[],
                 num_eval_episodes:int=5,
                 save_dir:Optional[str]=None,
                 save_interval:int=5000,
                 use_tf_functions:bool=True):
        # agent and environment parameters
        # `_agent_class` is an agent class instance containing the agent, 
        # actor-critic networks, and references to the environment while
        #  `_agent` is the actual agent.
        self._agent_class = agent
        self._agent = agent.agent
        # collection and evaluation environments
        self._collect_env = self._agent_class.collect_env
        self._eval_env = self._agent_class.eval_env
        # initialize the global training step
        self._train_step = self._agent_class.train_step
        # replay buffer params and replay init actor steps
        self._replay_buffer = replay_buffer
        # parameters differ for standard vs reverb buffers
        if 'reverb_buffer' in self._replay_buffer.__str__():
            self.use_reverb = True
            self._reverb_observer = self._replay_buffer._replay_observer
            self._experience_dataset_fn = (
                self._replay_buffer._experience_dataset_fn
            )
        else:
            self.use_reverb = False
        # instantiate drivers and collection steps
        # NOTE: drivers are defined when their relevant methods are called
        self._replay_exp_driver = None      # adds experience to replay buffer
        self._collect_driver = None         # collects training experience
        self.replay_init_steps = initial_replay_steps
        self.train_steps_per_iteration = train_steps_per_iteration
        self.replay_init_steps_per_run = self.replay_init_steps
        self.collection_steps_per_run = collection_steps_per_run
        self.num_eval_episodes = num_eval_episodes
        if self._agent._env_is_tf_env:
            self.train_metrics = [
                metrics.tf_metrics.NumberOfEpisodes(),
                metrics.tf_metrics.EnvironmentSteps(),
                metrics.tf_metrics.AverageReturnMetric(),
                metrics.tf_metrics.AverageEpisodeLengthMetric()
            ]
        else:
            self.train_metrics = [
                metrics.py_metrics.NumberOfEpisodes(),
                metrics.py_metrics.EnvironmentSteps(),
                metrics.py_metrics.AverageReturnMetric(),
                metrics.py_metrics.AverageEpisodeLengthMetric()
            ]
        if self._agent._env_is_tf_env:
            self.eval_metrics = [
                metrics.tf_metrics.AverageEpisodeLengthMetric(
                    buffer_size=self.num_eval_episodes
                ), 
                metrics.tf_metrics.AverageReturnMetric(
                buffer_size=self.num_eval_episodes
                )
            ] + add_eval_metrics
        else:
            self.eval_metrics = [
                metrics.py_metrics.AverageEpisodeLengthMetric(
                    buffer_size=self.num_eval_episodes
                ), 
                metrics.py_metrics.AverageReturnMetric(
                buffer_size=self.num_eval_episodes
                )
            ] + add_eval_metrics
        # tracking returns
        self._avg_return = None
        self._returns = []
        # policy saving
        if not save_dir:
            self.save_dir = tempfile.gettempdir()
        else:
            self.save_dir = save_dir
        self.save_interval = save_interval
        # set use of tf_functions
        self.use_tf_fn = use_tf_functions
        # initialize default policies
        # main policy:
        # self.eval_policy = policies.py_tf_eager_policy.PyTFEagerPolicy(
        #     self._agent.policy, use_tf_function=self.use_tf_fn
        # )
        self.eval_policy = self._agent.policy
        # main collection policy:
        # self.collect_policy = policies.py_tf_eager_policy.PyTFEagerPolicy(
        #     self._agent.collect_policy, use_tf_function=self.use_tf_fn
        # )
        self.collect_policy = self._agent.collect_policy
        # policy to populate replay memory:
        # self.replay_init_policy = policies.py_tf_eager_policy.PyTFEagerPolicy(
        #     self._agent.collect_policy, use_tf_function=self.use_tf_fn
        # )
        self.replay_init_policy = self._agent.collect_policy
    
    def set_replay_exp_driver(self, overwrite:bool=False):
        if (self._replay_exp_driver is not None) and (overwrite is False):
            return (
                "Replay experience driver already defined. Set `overwrite`" 
                + " to `True` to define a new one."
            )
        if self.use_reverb:
            replay_observer = self._reverb_observer
        else:
            replay_observer = self._replay_buffer.add_batch
        # define replay experience driver
        self._replay_exp_driver = dynamic_step_driver.DynamicStepDriver(
            env=self._collect_env, 
            policy=self.replay_init_policy,
            observers=[replay_observer], 
            num_steps=self.replay_init_steps_per_run
        )
        if self.use_tf_fn:
            self._replay_exp_driver.run = common.function(self._replay_exp_driver.run)

    def set_collect_driver(self, overwrite:bool=False):
        if (self._collect_driver is not None) and (overwrite is False):
            return (
                "Collect driver already defined. Set `overwrite`" 
                + " to `True` to define a new one."
            )
        if self.use_reverb:
            collect_observer = self._reverb_observer
        else:
            collect_observer = self._replay_buffer.add_batch
        # define main experience driver
        self._collect_driver = dynamic_step_driver.DynamicStepDriver(
            env=self._collect_env, 
            policy=self.collect_policy,
            observers=[collect_observer] + self.train_metrics, 
            num_steps=self.collection_steps_per_run
        )
        if self.use_tf_fn:
            self._collect_driver.run = common.function(self._collect_driver.run)

    def check_agent_setup(self):
        """Checks if actors and learner are set up before training."""
        if self._replay_buffer._replay_buffer is None:
            print(
                "Note: Replay buffer was configured but not initialized."
                + " Initializing now..."
                )
            self._replay_buffer.create_buffer()
        if self._replay_exp_driver is None:
            raise MissingDriverException(
                "A `replay_exp_driver` is required but none was found."
                + " Use `set_replay_exp_driver()` to define one."
            )
        if self._collect_driver is None:
            raise MissingDriverException(
                "A `collect_driver` is required but none was found."
                + " Use `set_collect_driver()` to define one."
            )

    def collect_replay_experience(self, show_progress_bar=False):
        """Collect seed steps for the replay memory."""
        self.check_agent_setup()
        # continue if setup is OK
        print('Collecting steps for replay memory...')
        if show_progress_bar:
            if (
                self.replay_init_steps >= 100 
                and self.replay_init_steps % 100 == 0
                ):
                self.replay_init_steps_per_run = self.replay_init_steps // 100
                if self.replay_init_steps_per_run >= 10:
                    self.replay_init_prog_bar = True
            else:
                print(
                    "\nInfo: Progress bar will not be shown."
                    + " It is only displayed when the total number of"
                    + " steps is 1000 or more and is divisible by 100."
                )
        if self.replay_init_prog_bar:
            with tqdm.trange(100) as prog_bar:
                for p in prog_bar:
                    self._replay_init_actor.run()
                    p_step = (p + 1) * self.replay_init_steps_per_run
                    prog_bar.set_description(f'step :{p_step}')
        else:
            self._replay_exp_driver.run()
        print('Done seeding replay memory.\n')
    
    def get_eval_metrics(self):
        """Run the eval policy and rertrieve training metrics."""
        results = metric_utils.eager_compute(
            metrics=self.eval_metrics,
            environment=self._eval_env,
            policy=self.eval_policy,
            num_episodes=self.num_eval_episodes,
            train_step=self._train_step
        )

        # return results
        # for metric in self._eval_actor.metrics:
        #     results[metric.name] = metric.result()
        
        return results

    def train_iteration(self):
        """
        Train the agent on a batch of samples from the replay memory.
        """
        # the second item in the tuple is sample info for debugging
        trajectories, _ = self._replay_buffer.iterate()
        return self._agent.train(experience=trajectories)

    @measure_run_time
    def train(self,
              num_train_iterations:int, 
              resume_from_prev_run:bool=True, 
              previous_train_step:Optional[int]=None, 
              run_replay_init:bool=False, 
              replay_init_prog_bar:bool=False, 
              eval_interval:Optional[int]=None, 
              log_interval:Optional[int]=None):
        """
        Train the agent.

        keyword args:
        ------------
        num_train_iterations: Number of training steps to run.
        resume_from_prev_run: resume training from the previous run.
        previous_train_step: select a spcific training step from the 
            previous run to restart from when `resume_from_prev_run` 
            is `True`. If `None`, the last recorded training step is 
            used.
        run_replay_init: Runs the `initial_collect_actor` to add 
            samples to the replay buffer without training. Typically 
            not needed if resuming training from a previous run.
        replay_init_prog_bar: show progress bar for replay init.
        learner_iterations: Number of iterations performed every time 
            the learner is called.
        eval_interval: Frequency of printing evaluation metrics. If 
            `None`, results are not printed.
        log_interval: Frequency of printing loss information. 
            If `None`, loss information is not printed.
        """
        self.check_agent_setup()

        if run_replay_init:
            self.collect_replay_experience(show_progress_bar=replay_init_prog_bar)
        if not resume_from_prev_run:
            # Reset training step counter
            self._train_step.assign(0)
        else:
            if previous_train_step is not None:
                self._train_step.assign(previous_train_step)

        if self.use_tf_fn:
            train_iteration = common.function(self.train_iteration)
        
        # set the time step for the collect driver to `None` to use current step
        time_step = None
        # Get initial state
        policy_state = self.collect_policy.get_initial_state(
            batch_size=self._collect_env.batch_size
        )

        # Get return values before start of training
        self._avg_return = self.get_eval_metrics().get('AverageReturn')
        self._returns.append(self._avg_return)

        # Run the training and evaluation loop
        print('Training...')
        for _ in range(num_train_iterations):
            # collect experience
            time_step, policy_state = self._collect_driver.run(
                time_step=time_step, policy_state=policy_state
            )
            for _ in self.train_steps_per_iteration:
                loss_info = self.train_iteration()
            # Print metrics and losses
            training_iteration = self._train_step.numpy()
            if eval_interval and training_iteration % eval_interval == 0:
                metrics = self.get_eval_metrics()
                print(
                    f'step {training_iteration}:'
                    + ', '.join(
                        [f'{name} = {res:.6f}' for name, res in metrics.items()]
                    )
                )
                # Keep track of evaluated returns
                self._returns.append(metrics.get('AverageReturn'))
            if log_interval and training_iteration % log_interval == 0:
                print(f'step {training_iteration}: loss = {loss_info.loss}')

        print('\nTraining finished.')