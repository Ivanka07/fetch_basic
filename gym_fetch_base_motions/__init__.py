import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='FetchBase-v0',
    entry_point='gym_fetch_base_motions.envs:FetchBaseEnv',
)
