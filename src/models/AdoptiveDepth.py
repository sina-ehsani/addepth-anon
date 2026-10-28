# Backwards-compatibility shim: this module was renamed to AdaptiveDepth.py
# (the old name was a typo). Import from .AdaptiveDepth in new code.
from .AdaptiveDepth import (  # noqa: F401
    TimeDistributedRL,
    AdDepth,
    QLearningAgent,
    translate_action_to_dynamic_layer_counts,
    get_initial_state,
    calculate_reward,
)
