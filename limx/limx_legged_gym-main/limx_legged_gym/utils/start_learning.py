from dataclasses import make_dataclass
from typing import Any
import copy
from limx_rl.runners import OnPolicyRunner
from limx_legged_gym.utils import helpers


def instance_to_dataclass(instance: Any) -> Any:
    """
    Dynamically creates a data class based on the attributes of an instance.
    Returns a new instance of this dynamically created data class.
    """
    cls = type(instance)
    attrs = vars(instance)
    DataClass = make_dataclass(cls.__name__ + 'DataClass', attrs.keys())
    return DataClass(**attrs)


def start_learning(runner: OnPolicyRunner, num_learning_iterations: int, init_at_random_ep_len: bool):
    try:
        old_env_cfg = copy.deepcopy(runner.env.cfg)
        runner.env.cfg = helpers.class_to_dict(runner.env.cfg)
        runner.learn(num_learning_iterations=num_learning_iterations, init_at_random_ep_len=init_at_random_ep_len)
    except:
        from limx_legged_gym.utils import instance_to_dataclass
        runner.env.cfg = instance_to_dataclass(old_env_cfg)
        runner.learn(num_learning_iterations=num_learning_iterations, init_at_random_ep_len=init_at_random_ep_len)
