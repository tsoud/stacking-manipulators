import pprint
from typing import Literal, Optional, Union

from tf_agents.replay_buffers import (
    tf_uniform_replay_buffer, py_uniform_replay_buffer
)
from tf_agents.trajectories import trajectory

# ----------------------------------------------------------------------------


class tfa_buffer:
    """
    Create a new reverb replay buffer.

    keyword args:
    ------------
    trajectory_data_spec: a tf_agents `Trajectory` object providing the  
        specification of the data collected and added to the buffer, 
        usually it is the agent's `.collect_data_spec` attribute.
    replay_max_size: maximum number of samples to hold in buffer. 
    buffer_type: use 'tf_uniform' to select TFUniformReplayBuffer or 
        'py_uniform' for PyUniformReplayBuffer. See:
        https://www.tensorflow.org/agents/api_docs/python/tf_agents/replay_buffers
        for details.
    insertion_batch_dim: batch dimension for trajectories inserted 
        into buffer, obtained from the environment's `env.batch_size`
        attribute. This arg is ignored when using the 'py_uniform' 
        buffer.
    timesteps_and_stride_per_sample: Tuple of (n_timesteps, stride).
        Number of timesteps to collect for each trajectory. The stride 
        describes how trajectories overlap. 
        Example: 
        (2, 1) describes a trajectory composed of a start state  
        and the next state. With stride == 1, the next state becomes 
        the start state for the subsequent trajectory.
        This parameter also specifies the `num_steps` argument for the 
        buffer's `as_dataset()` method.
    replay_scope_name: name to give replay buffer in the TensorFlow 
        computation graph.
    replay_sample_batch_size: number of samples to take from replay 
        memory for each training step. Samples are batched for input.
    num_parallel_calls: number of elements to process in parallel. If 
        `None`, use sequential processing.
    replay_sample_prefetch: number of replay dataset elements to 
        prefetch (cache) to speed up the pipeline.
    kwargs: optional kwargs for the replay buffer and replay dataset.
    """
    def __init__(self, 
                 trajectory_data_spec:trajectory.Trajectory,
                 replay_max_size:int,
                 buffer_type:Literal['tf_uniform', 'py_uniform']='tf_uniform',
                 insertion_batch_dim:Union[int, None]=None,
                 timesteps_and_stride_per_sample:tuple=(2,0),
                 replay_scope_name:Optional[str]=None,
                 replay_sample_batch_size:int=64,
                 num_parallel_calls:Optional[int]=4,
                 replay_sample_prefetch:int=10,
                 replay_buffer_kwargs:Optional[dict]=None,
                 replay_dataset_kwargs:Optional[dict]=None):

        self.trajectory_data_spec = trajectory_data_spec
        self.replay_max_size = replay_max_size
        self._buffer_type = buffer_type
        if self._buffer_type == 'tf_uniform':
            self._buffer_ctor = tf_uniform_replay_buffer.TFUniformReplayBuffer
            self.insertion_batch_dim = insertion_batch_dim
            if self.insertion_batch_dim is None:
                raise TypeError(
                    "`insertion_batch_dim` requires an integer argument when"
                    + f" `buffer_type = '{buffer_type}'` is selected. The" 
                    + " batch dim is given by the environment's `batch_size`" 
                    + " attribute."
                )
        elif self._buffer_type == 'py_uniform':
            self._buffer_ctor = py_uniform_replay_buffer.PyUniformReplayBuffer
        else:
            raise KeyError(
                f"`buffer_type = '{buffer_type}'` is invalid, valid options"
                + " are 'tf_uniform' or 'py_uniform'."
            )
        self.num_traj_steps = timesteps_and_stride_per_sample[0]
        self.traj_stride = (
            None if timesteps_and_stride_per_sample[1] == 0 
            else timesteps_and_stride_per_sample[1]
        )
        self.replay_scope_name = replay_scope_name
        self.replay_batch_size = replay_sample_batch_size
        self.num_parallel_calls = num_parallel_calls
        self.replay_prefetch_size = replay_sample_prefetch
        self.replay_buffer_kwargs = (
            {} if replay_buffer_kwargs is None else replay_buffer_kwargs
        )
        self.replay_dataset_kwargs = (
            {} if replay_dataset_kwargs is None else replay_dataset_kwargs
        )
        # the following attributes are created by the class methods
        self._replay_buffer = None
        self._replay_dataset = None
        self._replay_iter = None

    @staticmethod
    def add_optional_kwargs(main_kwargs, opt_kwargs):
        """
        Apply optional kwargs without overwriting instance attributes.
        """
        updated_kwargs = {
            k: v for k, v in opt_kwargs.items() if k not in main_kwargs.keys()
        }
        main_kwargs.update(updated_kwargs)

    def create_buffer(self):
        """
        Make a TF-Agents replay buffer from the instance configuration.
        """
        if self._buffer_type == 'tf_uniform':
            buffer_kwargs = dict(
                data_spec=self.trajectory_data_spec, 
                batch_size=self.replay_batch_size, 
                max_length=self.replay_max_size, 
                scope=self.replay_scope_name
            )
        else:
            buffer_kwargs = dict(
                data_spec=self.trajectory_data_spec, 
                max_length=self.replay_max_size, 
            )
        self.add_optional_kwargs(buffer_kwargs, self.replay_buffer_kwargs)
        self._replay_buffer = self._buffer_ctor(
            **buffer_kwargs
        )
        # make a training dataset and iterator from the replay buffer
        replay_dataset_kwargs = dict(
            sample_batch_size=self.replay_batch_size, 
            num_steps=self.num_traj_steps, 
            num_parallel_calls=self.num_parallel_calls
        )
        self.add_optional_kwargs(
            replay_dataset_kwargs, self.replay_dataset_kwargs
        )
        self._replay_dataset = self._replay_buffer.as_dataset(
            **replay_dataset_kwargs
        ).prefetch(self.replay_prefetch_size)
        self._replay_iter = iter(self._replay_dataset)
    
    def iterate(self):
        """Get a batch of trajectories from the replay buffer."""
        return next(self._replay_iter)

    def replay_buffer_info(self) -> dict:
        """
        Get information about the replay buffer and its current status.
        """
        info = {}
        if self._replay_buffer:
            info['data_spec'] = self._replay_buffer.data_spec
            info['max_size'] = self._replay_buffer.capacity.numpy()
            info['current_size'] = self._replay_buffer.num_frames().numpy()
        return pprint.pp(info, sort_dicts=False, indent=1, width=80)

    def clear_buffer(self):
        """Clears and resets the buffer."""
        self._replay_buffer.clear()