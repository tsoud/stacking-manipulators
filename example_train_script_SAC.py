import os
import warnings
from collections import namedtuple
from datetime import datetime
from functools import partial

import numpy as np
import pandas as pd
import tensorflow as tf
if tf.__version__ >= '2.0':
    gpu = tf.config.list_physical_devices('GPU')[0]
    tf.config.experimental.set_memory_growth(gpu, enable=True)
from tf_agents.environments import ParallelPyEnvironment

from robotic_stacking.bullet_envs import env_configs
from robotic_stacking.training import training_utils
from robotic_stacking.replay_buffers import reverb_buffer
from robotic_stacking.tfagents_envs import tfagents_stacking_env
from robotic_stacking.rl_agents.sac_agent import tfa_sac_agent
from robotic_stacking.training.sac_trainer import sac_trainer

def label_dir():
    now = datetime.now()
    return now.strftime("%b%d%y_%Hh%Mm")

warnings.filterwarnings(
    "ignore", 
    message='b3Warning[src/BulletInverseDynamics/MultiBodyTree.cpp,266]'
)

nn_params = training_utils.load_nn_params_from_json(
    'robotic_stacking/training/example_nn_params.json'
)
actor_net_params = nn_params.get('actor_net_params')
critic_net_params = nn_params.get('critic_net_params')

upper_b = np.array([0.01, 0.01, 0.01, 0.005*np.pi, 0.0015])
lower_b = -1*upper_b

target_pos = np.random.uniform([0.25, -0.4, 0.], [0.75, 0.4, 0.])
target_ort = np.random.uniform([0., 0., -0.5*np.pi], [0., 0., 0.5*np.pi])

test_env = env_configs.kvG3_stacking_5action(
    target_formation='default_pyramid',
    num_cubes=1,
    num_targets=1,
    target_formation_position=target_pos,
    target_cube_orientation=target_ort,
    n_transition_steps_per_sec=20,
    episode_time_limit=50,
    reset_on_episode_end=False,
    # use_GUI=True
).to_env()
# test_env.make()

# test_env2 = test_env.copy_env()
# test_env2.make()

tf_test_env = tfagents_stacking_env.tfagents_stacking_env(
    # env_configs.kvG3_stacking_5action,
    test_env,
    lower_b, upper_b,
    # config_kwargs=dict(
    #     # reward_function='sparse', 
    #     # use_GUI=True, 
    #     target_formation='default_pyramid',
    #     num_cubes=1,
    #     num_targets=1,
    #     target_formation_position=target_pos,
    #     target_cube_orientation=target_ort,
    #     n_transition_steps_per_sec=20,
    #     episode_time_limit=50,
    #     reset_on_episode_end=False,
    # )
)

# tf_test_env2 = tfagents_stacking_env.tfagents_stacking_env(
#     test_env2, 
#     lower_b, upper_b
# )

# tf_test_env2 = tf_test_env.copy_env()

# tf_test_env = test_env.wrap_to_TF_env()
# tf_test_env = test_env.wrap_to_TF_env(validation=3)

test_agent = tfa_sac_agent(
    tfa_env=tf_test_env, 
    # tfa_env=tf_test_env, 
    actor_net_params=actor_net_params, 
    critic_net_params=critic_net_params, 
)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

test_agent.make_agent(
    actor_optimizer=optimizer, 
    gamma=0.99, 
    # gamma=1.0, 
    # reward_scaling=1.0, 
    **{'gradient_clipping': 1.0}
)

# Training parameters
NUM_ITERATIONS = 250_000
INITIAL_REPLAY_STEPS = 5_000
REPLAY_MAX_SIZE = 1_000_000
REPLAY_BATCH_SIZE=256
REPLAY_PREFETCH=250
TRAJ_STEPS_AND_STRIDE = (2, 1)
NUM_EVAL_EPISODES = 5
EVAL_INTERVAL = 5_000
LOG_INTERVAL = 2_500
SAVE_INTERVAL = 5_000

NEW_DIR = True
# NEW_DIR = False

if NEW_DIR:
    SAVE_DIR = 'robotic_stacking/training/sac/runs/' + label_dir() + '/'
    if not os.path.isdir(SAVE_DIR):
        os.mkdir(SAVE_DIR)

# replay buffer
replay_mem = reverb_buffer.reverb_buffer(
    trajectory_data_spec=test_agent.collect_data_spec, 
    replay_max_size=REPLAY_MAX_SIZE, 
    replay_sample_batch_size=REPLAY_BATCH_SIZE,
    replay_sample_prefetch=REPLAY_PREFETCH, 
    timesteps_and_stride_per_sample=TRAJ_STEPS_AND_STRIDE, 
)
replay_mem.create_buffer()

# trainer
kv_stacking_sac_trainer = sac_trainer(
    agent=test_agent, 
    replay_buffer=replay_mem, 
    num_eval_episodes=NUM_EVAL_EPISODES, 
    initial_replay_steps=INITIAL_REPLAY_STEPS, 
    save_dir=SAVE_DIR, 
    save_interval=SAVE_INTERVAL
)

overwrite_existing = True #False
kv_stacking_sac_trainer.make_default_actors_and_learner(
    overwrite=overwrite_existing
)

kv_stacking_sac_trainer.train(
    num_train_iterations=NUM_ITERATIONS,
    # run_replay_init=True,
    run_replay_init=False,
    eval_interval=EVAL_INTERVAL,
    log_interval=LOG_INTERVAL,
)

# ----------------------------------------------------------------------------


results = namedtuple('run_info', 'step, returns')
RUN_ver = 3

run_returns = kv_stacking_sac_trainer._returns[:]
last_step = kv_stacking_sac_trainer.train_metrics[0].result()
run_returns = [('-1', r) if not isinstance(r, tuple) else r for r in run_returns]

run_name = 'run' + str(RUN_ver)
run_results = [results(r[0], r[1]) for r in run_returns]
run_results_df = pd.DataFrame(run_results)
run_results_df.to_csv(SAVE_DIR + run_name + '.csv')

# ----------------------------------------------------------------------------

from PIL import Image
eval_env = kv_stacking_sac_trainer._eval_env
eval_actor = kv_stacking_sac_trainer._eval_actor
frames = []
n_episodes = 3 #3
im_size = (640, 342)
# video_filename = 'kv_sac_trained_0.gif'
# video_filename = SAVE_DIR + 'kv_sac_trained_0.gif'
# video_filename = SAVE_DIR + 'kv_sac_trained_1.gif'
# video_filename = SAVE_DIR + 'kv_sac_trained_2.gif'
video_filename = SAVE_DIR + 'kv_sac_trained_3.gif'
# video_filename = 'kv_sac_trained_2.gif'
for _ in range(n_episodes):
    time_step = eval_env.reset()
    frame = eval_env.render(show=False, img_size=im_size)
    frames.append(frame)
    while not time_step.is_last():
        action_step = eval_actor.policy.action(time_step)
        time_step = eval_env.step(action_step.action)
        frame = eval_env.render(show=False, img_size=im_size)
        frames.append(frame)
with open(video_filename, 'wb') as fp:
    frames[0].save(fp, format='GIF', append_images=frames[1:], 
        save_all=True, optimize=True, duration=20)

# import inspect
# class hello:

#     def __init__(self, a, b, c, d=None):
#         self.a_arg = a
#         self.b_arg = b
#         self.c_arg = c
#         self.d_arg = d
#         # params, args = inspect.signature(self.__init__).parameters,  locals()
#         # self._init_params = {k: v for k, v in zip(params.keys(), args)}
#         self.__init_args = locals().copy()
#         self._params = {k: v for k, v in self.__init_args.items() if not k == 'self'}

#     def show(self):
#         print(f'a = {self.a_arg}, b = {self.b_arg}, c = {self.c_arg}, d = {self.d_arg}')

#     def make_copy(self):
#         new_copy = hello(**self._params)
#         new_copy.__dict__.update(self.__dict__)
#         return new_copy