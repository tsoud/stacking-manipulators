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
from tf_agents.replay_buffers import reverb_replay_buffer, reverb_utils
from tf_agents.train import actor, learner, triggers
from tf_agents.train.utils import spec_utils, strategy_utils
from tf_agents.train.utils import train_utils as tfa_train_utils
from tf_agents.typing import types as tfa_types
from tf_agents.utils import common


from robotic_stacking.replay_buffers import reverb_buffer
from robotic_stacking.rl_agents import sac_agent
from robotic_stacking.training.training_utils import (
    NoReplayBufferException, 
    MissingActorOrLearnerException, 
    measure_run_time
)

# ----------------------------------------------------------------------------


class sac_trainer:
    """
    Configure and run training with a Soft Actor-Critic model.

    A trainer is initialized with a replay buffer and three policies: 
    - the main policy being trained (the `eval_policy`), 
    - a second policy for data collection (the `collect_policy`),
    - a random policy to collect seed samples for replay memory.
    Each policy is run by a separate Actor worker. A Learner uses the 
    replay experience to apply gradient updates to policy variables.

    keyword args:
    ------------
    agent_class: Instance of `tfa_sac_agent` containing the agent 
        to train.
    replay_buffer: An instance of `reverb_buffer` or any buffer 
        from `tf_agents.replay_buffers.replay_buffer.ReplayBuffer`
    initial_replay_steps: Number of steps for collecting initial 
        samples for the replay memory before training begins.
    num_eval_episodes: Number of episodes to aggregate for evaluation.
    collection_steps_per_run: Number of environment steps to iterate 
        over when evaluating the main policy actor.
    train_metrics: Additional metrics for training and evaluation.
        By default, training monitors global step size, loss, average 
        episode length and average reward.
    save_dir: Directory to save the model. If `None`, the model is 
        saved to a `tempdir` object.
    save_interval: How often to save the policy.
    """
    def __init__(self,  
                 agent:sac_agent.tfa_sac_agent, 
                 replay_buffer:
                    Union[reverb_buffer.reverb_buffer, replay_buffers.replay_buffer.ReplayBuffer], 
                 initial_replay_steps:int, 
                 num_eval_episodes:int=5, 
                 collection_steps_per_run:int=1, 
                 train_metrics:list=[], 
                 save_dir:Optional[str]=None, 
                 save_interval:int=5000):
        # agent and environment parameters
        # `_agent_class` is an agent class instance containing the agent, 
        # actor-critic networks, and references to the environment while
        #  `_agent` is the actual agent.
        self._agent_class = agent
        self._agent = agent.agent
        self._collect_env = self._agent_class.collect_env
        self._eval_env = self._agent_class.eval_env
        self._train_step = self._agent_class.train_step
        # replay buffer params and replay init actor steps
        self._replay_buffer = replay_buffer  
        self._replay_observer = self._replay_buffer._replay_observer
        self._experience_dataset_fn = (
            self._replay_buffer._experience_dataset_fn
        )
        self.replay_init_steps = initial_replay_steps
        # for metering the initial replay steps:
        self.replay_init_prog_bar = False
        if self.replay_init_steps >= 100 and self.replay_init_steps % 100 == 0:
            self.replay_init_steps_per_run = self.replay_init_steps // 100
            if self.replay_init_steps_per_run >= 10:
                self.replay_init_prog_bar = True
        else:
            self.replay_init_steps_per_run = self.replay_init_steps
        # metrics
        self.num_eval_episodes = num_eval_episodes
        self.collection_steps_per_run = collection_steps_per_run
        self.env_step = metrics.py_metrics.EnvironmentSteps()
        if self.env_step not in train_metrics:
            train_metrics.append(self.env_step)
        self.train_metrics = train_metrics
        # tracking returns
        self._avg_return = None
        self._returns = []
        # policy saving
        if not save_dir:
            self.save_dir = tempfile.gettempdir()
        else:
            self.save_dir = save_dir
        self.save_interval = save_interval
        # initialize default policies
        # main policy:
        self.eval_policy = policies.py_tf_eager_policy.PyTFEagerPolicy(
            self._agent.policy, use_tf_function=True
        )
        # collection policy:
        self.collect_policy = policies.py_tf_eager_policy.PyTFEagerPolicy(
            self._agent.collect_policy, use_tf_function=True
        )
        # random policy for replay initialization:
        self.replay_init_policy = policies.random_py_policy.RandomPyPolicy(
            time_step_spec=self._collect_env.time_step_spec(), 
            action_spec=self._collect_env.action_spec()
        )
        # instantiate default actor variables
        self._replay_init_actor = None          # seeds replay buffer
        self._collect_actor = None              # collects training experience
        self._eval_actor = None                 # evaluates policy
        # get distribution strategy from the agent class
        self._strategy = self._agent_class.strategy
        # create learner
        self._learner_triggers = None
        self._learner = None
        # useful info about actor and learner configs
        self._actor_learner_info = {}

    def actor_and_learner_info(self):
        """Get a description of the actors and learner."""
        return pprint.pp(self._actor_learner_info, indent=1, width=120)

    def create_replay_init_actor(self, overwrite:bool=False, **kwargs):
        """
        Create an actor worker to seed the replay buffer.

        keyword args:
        ------------
        overwrite: overwrite previous actor if one already exists.
        kwargs: optional kwargs to pass to the Actor. Optional kwargs 
            will not overwrite args defined by instance attributes.
        """
        if self._replay_init_actor is None or overwrite:
            actor_args = dict(
                env=self._collect_env,
                policy=self.replay_init_policy,
                train_step=self._train_step,
                steps_per_run=self.replay_init_steps_per_run,
                # steps_per_run=self.replay_init_steps,
                observers=[self._replay_observer]
            )
            actor_args.update(
                {k: v for k, v in kwargs.items() if k not in actor_args.keys()}
            )
            self._actor_learner_info['replay_init_actor'] = actor_args
            self._replay_init_actor = actor.Actor(**actor_args)              

    def create_collect_actor(self, overwrite:bool=False, **kwargs):
        """
        Create an actor worker to run the collect policy.

        keyword args:
        ------------
        overwrite: overwrite previous actor if one already exists.
        kwargs: optional kwargs to pass to the Actor. Optional kwargs 
            will not overwrite args defined by instance attributes.
        """
        if self._collect_actor is None or overwrite:
            actor_args = dict(
                env=self._collect_env,
                policy=self.collect_policy,
                train_step=self._train_step,
                steps_per_run=self.collection_steps_per_run,
                metrics=actor.collect_metrics(buffer_size=10),
                summary_dir=os.path.join(self.save_dir, learner.TRAIN_DIR),
                observers=[self._replay_observer, self.env_step]
            )
            actor_args.update(
                {k: v for k, v in kwargs.items() if k not in actor_args.keys()}
            )
            self._actor_learner_info['collect_actor'] = actor_args
            self._collect_actor = actor.Actor(**actor_args)

    def create_eval_actor(self, overwrite:bool=False, **kwargs):
        """
        Create an actor worker to run the main (eval) policy.

        keyword args:
        ------------
        overwrite: overwrite previous actor if one already exists.
        kwargs: optional kwargs to pass to the Actor. Optional kwargs 
            will not overwrite args defined by instance attributes.
        """
        if self._eval_actor is None or overwrite:
            actor_args = dict(
                env=self._eval_env,
                policy=self.eval_policy,
                train_step=self._train_step,
                episodes_per_run=self.num_eval_episodes,
                metrics=actor.eval_metrics(buffer_size=self.num_eval_episodes),
                summary_dir=os.path.join(self.save_dir, 'eval'),
            )
            actor_args.update(
                {k: v for k, v in kwargs.items() if k not in actor_args.keys()}
            )
            self._actor_learner_info['eval_actor'] = actor_args
            self._eval_actor = actor.Actor(**actor_args)

    def set_learner_triggers(self, 
                             add_triggers:Optional[list]=None, 
                             overwrite_defaults:bool=False, 
                             overwrite_existing:bool=False):
        """
        Define list of trigger functions for the learner

        Triggers are called after each given interval of calls to the 
        learner to save a checkpoint or log information. See 
        www.tensorflow.org/agents/api_docs/python/tf_agents/train/triggers 
        for more details.
        By default, two triggers `PolicySavedModelTrigger` and 
        `StepPerSecondLogTrigger` are created with pre-configured args.

        add_triggers: additional triggers to add to defaults, or a list 
            of new triggers to replace them.
        overwrite_defaults: overwrites the default list of triggers 
            with the list in `add_triggers`.
        overwrite_existing: overwrite any previously defined triggers.
        """
        if self._learner_triggers is None or overwrite_existing:
            default_triggers = [
                triggers.PolicySavedModelTrigger(
                    saved_model_dir=os.path.join(
                        self.save_dir, learner.POLICY_SAVED_MODEL_DIR
                    ), 
                    agent=self._agent, 
                    train_step=self._train_step, 
                    interval=self.save_interval
                ), 
                triggers.StepPerSecondLogTrigger(
                    train_step=self._train_step, 
                    interval=self.save_interval // 5
                )
            ]
            if add_triggers is None:
                self._learner_triggers = default_triggers.copy()
            else:
                if not isinstance(add_triggers, list):
                    raise TypeError('`new_triggers` must be a list.')
                for trigger in add_triggers:
                    if overwrite_defaults:
                        if type(trigger) is type(default_triggers[0]):
                            default_triggers[0] = trigger
                        elif type(trigger) is type(default_triggers[1]):
                            default_triggers[1] = trigger
                        else:
                            default_triggers.append(trigger)
                    else:
                        if type(trigger) in (
                            type(default_triggers[0]), 
                            type(default_triggers[1])
                            ):
                            # skip if trigger in defaults
                            continue
                        else:
                            default_triggers.append(trigger)
                self._learner_triggers = default_triggers.copy()

    def create_learner(self, overwrite:bool=False, **kwargs):
        """
        Create a Learner to train the networks.

        keyword args:
        ------------
        overwrite: overwrite previous learner if one already exists.
        kwargs: optional kwargs to pass to the Learner. Optional kwargs 
            will not overwrite args defined by instance attributes.
        """
        if self._learner_triggers is None:
            print(
                'Info: no triggers were defined for Learner. ' 
                + 'Proceeding with default triggers.'
            )
            self.set_learner_triggers()
        if self._learner is None or overwrite:
            learner_args = dict(
                root_dir=os.path.join(
                    self.save_dir, learner.POLICY_SAVED_MODEL_DIR
                ), 
                train_step=self._train_step, 
                agent=self._agent, 
                experience_dataset_fn=self._experience_dataset_fn, 
                triggers=self._learner_triggers, 
                strategy=self._strategy
            )
            learner_args.update(
                {k: v for k, v in kwargs.items() if k not in learner_args.keys()}
            )
            self._actor_learner_info['learner'] = learner_args
            self._learner = learner.Learner(**learner_args)

    def make_default_actors_and_learner(self, overwrite:bool=False):
        """
        Create all actors and learner with default arguments.

        keyword args:
        ------------
        overwrite: overwrite any exising actors or learner.
        """
        print('Creating actors and learner with default args.\n')
        self.create_replay_init_actor(overwrite=overwrite)
        self.create_collect_actor(overwrite=overwrite)
        self.create_eval_actor(overwrite=overwrite)
        self.set_learner_triggers(overwrite_existing=overwrite)
        self.create_learner(overwrite=overwrite)
        print(
            '\nDone. Use `actor_and_learner_info()` to view configuration.'
            + ' To create actors or learner with optional arguments, call the'
            + ' relevant actor/learner creation method directly.\n'
        )

    def check_agent_setup(self):
        """Checks if actors and learner are set up before training."""
        if self._replay_buffer._replay_buffer is None:
            print(
                "Note: Replay buffer was configured but not initialized." 
                + " Initializing now..."
                )
            self._replay_buffer.create_buffer()
        if self._replay_init_actor is None:
            raise MissingActorOrLearnerException(
                'A `replay_init_actor` is required but none was found. ' 
                + 'Use `create_replay_init_actor()` to define one.'
            )
        if self._collect_actor is None:
            raise MissingActorOrLearnerException(
                'A `collect_actor` is required but none was found. ' 
                + 'Use `create_collect_actor()` to define one.'
            )
        if self._eval_actor is None:
            raise MissingActorOrLearnerException(
                'An `eval_actor` is required but none was found. ' 
                + 'Use `create_eval_actor()` to define one.'
            )
        if self._learner is None:
            raise MissingActorOrLearnerException(
                'A `learner` is required but none was found. ' 
                + 'Use `create_learner()` to define one.'
            )

    def run_replay_collector(self, show_progress_bar=True):
        """Collect seed steps for the replay memory."""
        self.check_agent_setup()

        show_prog = (
            '' if self.replay_init_prog_bar
            else (
                '\nInfo: Progress bar will not be shown. It is only displayed'
                + ' when the total number of steps is 1000 or more and is'
                + ' divisible by 100.'
            )
        )
        print('Collecting steps for replay memory...' + show_prog)
        if self.replay_init_prog_bar and show_progress_bar:
            with tqdm.trange(100) as prog_bar:
                for p in prog_bar:
                    self._replay_init_actor.run()
                    p_step = (p + 1) * self.replay_init_steps_per_run
                    prog_bar.set_description(f'step :{p_step}')
        else:
            self._replay_init_actor.run()
        print('Done seeding replay memory.\n')
    
    def get_eval_metrics(self):
        """Run the eval actor and rertrieve training metrics."""
        self._eval_actor.run()
        results = {}
        for metric in self._eval_actor.metrics:
            results[metric.name] = metric.result()
        
        return results

    @measure_run_time
    def train(self,
              num_train_iterations:int, 
              resume_from_prev_run:bool=True, 
              previous_train_step:Optional[int]=None, 
              run_replay_init:bool=False, 
              replay_init_prog_bar:bool=True, 
              learner_iterations:int=1, 
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
            self.run_replay_collector(show_progress_bar=replay_init_prog_bar)
        # Reset training step counter
        if not resume_from_prev_run:
            self._agent.train_step_counter.assign(0)
        else:
            if previous_train_step is not None:
                self._agent.train_step_counter.assign(previous_train_step)
        # Get return values before start of training
        self._avg_return = self.get_eval_metrics().get('AverageReturn')
        self._returns.append(self._avg_return)

        # Run the training and evaluation loop
        print('Training...')
        for _ in range(num_train_iterations):
            # Train
            self._collect_actor.run()
            loss_info = self._learner.run(iterations=learner_iterations)
            # Print metrics and losses
            step = self._learner.train_step_numpy
            if eval_interval and step % eval_interval == 0:
                metrics = self.get_eval_metrics()
                print(f'step {step}:', 
                    ', '.join(
                    [f'{name} = {res:.6f}' for name, res in metrics.items()]
                    )
                )
                # Keep track of evaluated returns
                self._returns.append((step, metrics.get('AverageReturn')))
            if log_interval and step % log_interval == 0:
                print(f'step {step}: loss = {loss_info.loss.numpy()}')

        print('\nTraining finished.')