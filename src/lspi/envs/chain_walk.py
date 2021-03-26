import gym
import numpy as np
from gym.envs.toy_text import discrete


class ChainWalkEnv(discrete.DiscreteEnv):
    """Chain Walk environment
    There are two actions available, “left” (L) and “right” (R). The actions succeed
    with probability 0.9, changing the state in the intended direction, and fail with 
    probability 0.1, changing the state in the opposite direction; the two boundaries 
    of the chain are dead-ends.
    
    Least-Squares Policy Iteration (Lagoudakis and Parr, 2003)
    http://www.jmlr.org/papers/volume4/lagoudakis03a/lagoudakis03a.pdf
    Policy Iteration for Factored MDPs (Koller and Parr, 2000)
    https://arxiv.org/pdf/1301.3869.pdf
    """
    def __init__(self, nS=4, slip=0.1, reward_function=None):
        """
        Args:
            nS (int, optional): number of states. Defaults to 4.
            slip (float, optional): slipping probability. Defaults to 0.1.
            reward_function (optional): state-reward function. Defaults to None.

        Returns:
            gym.Env: Chain Walk environment
        """
        nA = 2
        if reward_function is None:

            def reward_function(s):
                if s in [0, nS - 1]:
                    return 0
                else:
                    return 1

        P = {
            s: {
                a: [(1 - slip, np.clip(s + 2 * a - 1, 0,
                                       nS - 1), reward_function(s), False),
                    (slip, np.clip(s - 2 * a + 1, 0,
                                   nS - 1), reward_function(s), False)]
                for a in range(nA)
            }
            for s in range(nS)
        }
        isd = 1 / nS * np.ones(nS)
        super(ChainWalkEnv, self).__init__(nS, nA, P, isd)
