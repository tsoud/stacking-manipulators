import pprint
from typing import Literal, Optional

import reverb
from tf_agents.replay_buffers import reverb_replay_buffer, reverb_utils

# ----------------------------------------------------------------------------


class reverb_buffer:
    """
    Create a new reverb replay buffer.

    keyword args:
    ------------
    replay_max_size: Maximum number of samples to hold in buffer. 
    timesteps_and_stride_per_sample: Tuple of (n_timesteps, stride).
        Number of timesteps to collect for each trajectory. The stride 
        describes how trajectories overlap. 
        Example: 
        (2, 1) describes a trajectory composed of a start state  
        and the next state. With stride == 1, the next state becomes 
        the start state for the subsequent trajectory.
        This parameter also specifies the `num_steps` argument for the 
        buffer's `as_dataset()` method.
    replay_name: Name to give replay buffer table. If `None`, the table 
        is named after the replay sample e.g. 'uniform_sampling_table'.
    replay_sampler: Method for sampling from replay buffer. Call 
        `selector_types(print_only=True)` to see available selection 
        methods.
    replay_remover: Method for removing samples from replay buffer. 
        Call `selector_types(print_only=True)` to see available 
        selection methods.
    replay_rate_limiter: Method to call for organizing insertion into 
        and sampling from replay buffer. See 
        https://github.com/deepmind/reverb#rate-limiting for details.
    replay_sample_batch_size: Number of samples to take from replay 
        memory for each training step. Samples are batched for input.
    replay_sample_prefetch: Number of replay dataset elements to 
        prefetch (cache) to speed up the pipeline.
    priority_exp: Priority exponent to use with prioritized sampler. 
        Ignored for other buffer types.
    kwargs: optional kwargs for the reverb table, reverb buffer, 
        replay observer, and replay dataset.
    """
    def __init__(self, 
                 replay_max_size:int, 
                 timesteps_and_stride_per_sample:tuple, 
                 replay_name:Optional[str]=None, 
                 replay_sampler:Literal[
                        'Fifo',
                        'Lifo',
                        'MaxHeap',
                        'MinHeap',
                        'Prioritized',
                        'Uniform'
                        ]='Uniform',
                 replay_remover:Literal[
                        'Fifo',
                        'Lifo',
                        'MaxHeap',
                        'MinHeap',
                        'Prioritized',
                        'Uniform'
                        ]='Fifo', 
                 replay_rate_limiter:
                    Optional[reverb.rate_limiters.RateLimiter]=None, 
                 replay_sample_batch_size:int=64, 
                 replay_sample_prefetch:int=50, 
                 priority_exp:float=0.8, 
                 reverb_table_kwargs:Optional[dict]=None, 
                 reverb_buffer_kwargs:Optional[dict]=None, 
                 replay_observer_kwargs:Optional[dict]=None, 
                 replay_dataset_kwargs:Optional[dict]=None):
        # reverb offers different methods for adding and removing buffer items
        self._reverb_selectors = {
            'Fifo': reverb.selectors.Fifo(), 
            'Lifo': reverb.selectors.Lifo(), 
            'MaxHeap': reverb.selectors.MaxHeap(), 
            'MinHeap': reverb.selectors.MinHeap(), 
            'Prioritized': reverb.selectors.Prioritized(priority_exp), 
            'Uniform': reverb.selectors.Uniform()
        }
        # main instance attributes
        self.replay_max_size = replay_max_size
        self.num_traj_steps = timesteps_and_stride_per_sample[0]
        self.traj_stride = timesteps_and_stride_per_sample[1]
        if replay_name:
            self.table_name = replay_name
        else:
            self.table_name = replay_sampler.lower() + '_sampling_table'
        self.replay_sampler = self._reverb_selectors.get(replay_sampler)
        self.replay_remover = self._reverb_selectors.get(replay_remover)
        self.replay_rate_limiter = (
            replay_rate_limiter or reverb.rate_limiters.MinSize(1)
        )
        self.replay_batch_size = replay_sample_batch_size
        self.replay_prefetch_size = replay_sample_prefetch
        # optional kwargs passed to buffer methods
        self.reverb_table_kwargs = (
            {} if reverb_table_kwargs is None else reverb_table_kwargs
        )
        self.reverb_buffer_kwargs = (
            {} if reverb_buffer_kwargs is None else reverb_buffer_kwargs
        )
        self.replay_observer_kwargs = (
            {} if replay_observer_kwargs is None else replay_observer_kwargs
        )
        self.replay_dataset_kwargs = (
            {} if replay_dataset_kwargs is None else replay_dataset_kwargs
        )
        # the following attributes are created by the class methods
        self._replay_buffer = None
        self._reverb_table = None
        self._reverb_server = None
        self._replay_observer = None
        self._replay_dataset = None

    @staticmethod
    def add_optional_kwargs(main_kwargs, opt_kwargs):
        """
        Apply optional kwargs without overwriting instance attributes.
        """
        updated_kwargs = {
            k: v for k, v in opt_kwargs.items() if k not in main_kwargs.keys()
        }
        return main_kwargs.update(updated_kwargs)

    def show_selector_options(self):
        """
        Print selector and remover options for the reverb buffer.
        """
        print(*[s for s in self._reverb_selectors.keys()], sep='\n')

    def create_buffer(self):
        """
        Make a reverb replay buffer from the instance configuration.
        """
        # the reverb table determines how trajectories are added to 
        # and removed from the reverb replay buffer
        main_table_kwargs = dict(
            name=self.table_name, 
            max_size=self.replay_max_size, 
            sampler=self.replay_sampler, 
            remover=self.replay_remover, 
            rate_limiter=self.replay_rate_limiter
        )   # args defined by class constructor
        table_kwargs = self.add_optional_kwargs(
            main_table_kwargs, self.reverb_table_kwargs
        )
        self._reverb_table = reverb.Table(**table_kwargs)
        # the reverb replay buffer is set up as a server and client
        self._reverb_server = reverb.Server([self._reverb_table])
        main_buffer_kwargs = dict(
            py_client=self._replay_buffer.py_client, 
            table_name=self.table_name, 
            sequence_length=self.num_traj_steps, 
            stride_length=self.traj_stride
        )
        buffer_kwargs = self.add_optional_kwargs(
            main_buffer_kwargs, self.reverb_buffer_kwargs
        )
        self._replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
            **buffer_kwargs
        )
        # the replay observer writes trajectories from actors to the buffer
        main_observer_kwargs = dict(
            py_client=self._replay_buffer.py_client, 
            table_name=self.table_name, 
            sequence_length=self.num_traj_steps, 
            stride_length=self.traj_stride
        )
        observer_kwargs = self.add_optional_kwargs(
            main_observer_kwargs, self.replay_observer_kwargs
        )
        self._replay_observer = reverb_utils.ReverbAddTrajectoryObserver(
            **observer_kwargs
        )
        # a training dataset passes trajectories from the replay buffer 
        # to the learner
        main_replay_dataset_kwargs = dict(
            sample_batch_size=self.replay_batch_size, 
            num_steps=self.num_traj_steps
        )
        replay_dataset_kwargs = self.add_optional_kwargs(
            main_replay_dataset_kwargs, self.replay_dataset_kwargs
        )
        self._replay_dataset = self._replay_buffer.as_dataset(
            **replay_dataset_kwargs
        )#.prefetch(self.replay_prefetch_size)
        self._experience_dataset_fn = lambda: self._replay_dataset

    def replay_buffer_info(self) -> dict:
        """
        Get information about the replay buffer and its current status.
        """
        info = {}
        max_size = self._replay_buffer.capacity
        current_size = self._replay_buffer.get_table_info().current_size
        if self._replay_buffer:
            info['table_name'] = self._replay_buffer.get_table_info().name
            info['current_size'] = f'{current_size} current / {max_size} max'
            info['server_port'] = self._replay_buffer.local_server.port
            info['sampler'] = self._replay_buffer.get_table_info()\
                .sampler_options
            info['remover'] = self._replay_buffer.get_table_info()\
                .remover_options
            info['rate_limiter_info'] = self.replay_rate_limiter.__repr__()
            info['sampling_batch_size'] = self.replay_batch_size
            info['prefetch_limit'] = self.replay_prefetch_size
            info['max_times_sampled'] = self._replay_buffer.get_table_info()\
                .max_times_sampled
        return pprint.pp(info, sort_dicts=False, indent=1, width=80)

