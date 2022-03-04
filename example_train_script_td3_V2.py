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
from tf_agents.specs import tensor_spec

from robotic_stacking.bullet_envs import env_configs
from robotic_stacking.training import training_utils
from robotic_stacking.replay_buffers import reverb_buffer, tfa_buffer
from robotic_stacking.tfagents_envs import tfagents_stacking_env

from robotic_stacking.rl_agents.td3_agent_v2 import tfa_td3_agent
from robotic_stacking.training.td3_trainer_ActorLearner import td3_trainer_AL


def label_dir():
    now = datetime.now()
    return now.strftime("%b%d%y_%Hh%Mm")

warnings.filterwarnings(
    "ignore", 
    message='b3Warning[src/BulletInverseDynamics/MultiBodyTree.cpp,266]'
)

nn_params = training_utils.load_nn_params_from_json(
    'robotic_stacking/training/example_nn_params_td3.json'
)
actor_net_params = nn_params.get('actor_net_params')
critic_net_params = nn_params.get('critic_net_params')

upper_b = np.array([0.01, 0.01, 0.01, 0.005*np.pi, 0.0015])
lower_b = -1*upper_b
# ran
target_pos = np.random.uniform([0.25, -0.25, 0.], [0.75, 0.25, 0.])
target_ort = np.random.uniform([0., 0., -0.5*np.pi], [0., 0., 0.5*np.pi])

test_env = env_configs.kvG3_stacking_5action(
    target_formation='default_pyramid',
    num_cubes=1,
    num_targets=1,
    target_formation_position=target_pos,
    target_cube_orientation=target_ort,
    reward_function='dense',
    n_transition_steps_per_sec=20,
    episode_time_limit=50,
    reset_on_episode_end=False,
    # use_GUI=True
).to_env()

tf_test_env = tfagents_stacking_env.tfagents_stacking_env(
    test_env,
    lower_b, upper_b,
)


test_agent = tfa_td3_agent(
    tfa_env=tf_test_env,
    actor_net_params=actor_net_params,
    critic_net_params=critic_net_params,
    use_gpu=False,
)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

test_agent.make_agent(
    actor_optimizer=optimizer,
    gamma=0.99,
    exploration_stddev=0.5
    # reward_scaling=1.0, 
    # **{'gradient_clipping': 1.0}
)

# Training parameters
NUM_ITERATIONS = 500_000
INITIAL_REPLAY_STEPS = 50_000
REPLAY_MAX_SIZE = 1_000_000
RAND_WARMUP = False #True
REPLAY_BATCH_SIZE = 256
REPLAY_PREFETCH = 20
TRAJ_STEPS_AND_STRIDE = (2, 1)
NUM_EVAL_EPISODES = 5
EVAL_INTERVAL = 5_000
LOG_INTERVAL = 2_500
SAVE_INTERVAL = 5_000
CKPT_INTERVAL = 10_000

NEW_SAVE_DIR = True #False

if NEW_SAVE_DIR:
    SAVE_DIR = 'robotic_stacking/training/td3/runs/' + label_dir() + '/'
if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)


# replay_buffer_signature = tensor_spec.from_spec(test_agent.collect_data_spec)
# replay_buffer_signature = tensor_spec.add_outer_dims_nest(
#     replay_buffer_signature, (None, )
# )
# reverb buffer
replay_mem_rvrb = reverb_buffer.reverb_buffer(
    trajectory_data_spec=test_agent.collect_data_spec, 
    replay_max_size=REPLAY_MAX_SIZE, 
    replay_sample_batch_size=REPLAY_BATCH_SIZE,
    replay_sample_prefetch=REPLAY_PREFETCH, 
    timesteps_and_stride_per_sample=TRAJ_STEPS_AND_STRIDE,
    # reverb_table_kwargs={
    #     'signature': replay_buffer_signature
    # }
)
replay_mem_rvrb.create_buffer()


# trainer 2
kv_stacking_td3_trainer = td3_trainer_AL(
    agent=test_agent,
    replay_buffer=replay_mem_rvrb,
    random_warmup_policy=RAND_WARMUP,
    num_eval_episodes=NUM_EVAL_EPISODES,
    initial_replay_steps=INITIAL_REPLAY_STEPS,
    save_dir=SAVE_DIR,
    save_interval=SAVE_INTERVAL, 
)

kv_stacking_td3_trainer.make_default_actors_and_learner(
    overwrite=True
)

# kv_stacking_td3_trainer.run_replay_collector()

kv_stacking_td3_trainer.train(
    num_train_iterations=NUM_ITERATIONS,
    run_replay_init=True, 
    # run_replay_init=False, 
    eval_interval=EVAL_INTERVAL,
    log_interval=LOG_INTERVAL,
)


# ----------------------------------------------------------------------------

results = namedtuple('run_info', 'step, returns, TD_error_loss')
RUN_ver = 0

run_returns = kv_stacking_td3_trainer._returns[:]
last_step = kv_stacking_td3_trainer.train_metrics[0].result()
run_returns = [
    ('', r, '') if not isinstance(r, tuple) else r for r in run_returns
]

run_name = 'run' + str(RUN_ver)
run_results = [results(r[0], r[1], r[2]) for r in run_returns]
run_results_df = pd.DataFrame(run_results)
run_results_df.to_csv(SAVE_DIR + run_name + '.csv')

# ----------------------------------------------------------------------------

from PIL import Image
eval_env = kv_stacking_td3_trainer._eval_env
eval_actor = kv_stacking_td3_trainer._eval_actor
frames = []
n_episodes = 3
im_size = (640, 342)
video_filename = SAVE_DIR + 'kv_td3_trained_0.gif'
# video_filename = SAVE_DIR + 'kv_td3_trained_1.gif'
# video_filename = SAVE_DIR + 'kv_td3_trained_2.gif'
# video_filename = SAVE_DIR + 'kv_td3_trained_3.gif'
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
