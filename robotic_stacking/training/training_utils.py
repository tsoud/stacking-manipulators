import json
import time
from datetime import timedelta
from functools import wraps

# --------------------------------------------------------------------------- #
# Some helpful tools for RL training
# --------------------------------------------------------------------------- #



class NoReplayBufferException(Exception):
    """Raise an exception if the replay buffer is not defined."""
    pass

class MissingActorOrLearnerException(Exception):
    """Raise an exception if an actor or learner is not defined."""
    pass

# --------------------------------------------------------------------------- #


def measure_run_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t_start = time.perf_counter()
        func(*args, **kwargs)
        run_time = str(timedelta(seconds=(time.perf_counter() - t_start)))
        print(f'\nRun time: {run_time}')
    return wrapper


def load_nn_params_from_json(filepath):
    with open(filepath, 'r') as fp:
        params = json.load(fp)
    return params

# --------------------------------------------------------------------------- #