import math
import random

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding


class InvertedPendulumEnv(gym.Env):
    """Inverted Pendulum environment
    The inverted pendulum problem requires balancing a pendulum of unknown length and 
    mass at the upright position by applying forces to the cart it is attached to.
    
    Least-Squares Policy Iteration (Lagoudakis and Parr, 2003)
    http://www.jmlr.org/papers/volume4/lagoudakis03a/lagoudakis03a.pdf
    An approach to fuzzy control of nonlinear systems: Stability and designissues (Wang et al., 1996)
    https://ieeexplore.ieee.org/document/481841/
    
    Adapted from : 
    https://github.com/TylerGoeringer/PyPendulum
    """
    def __init__(self,
                 nA=3,
                 f=50,
                 f_noise=10,
                 m=2.0,
                 M=8.0,
                 l=0.5,
                 g=9.8,
                 dt=0.1):
        """
        Args:
            nA (int, optional): number of discrete actions. Defaults to None.
            f (int, optional): force range. Defaults to 50.
            f_noise (int, optional): force noise. Defaults to 10.
            m (float, optional): mass of the pendulum. Defaults to 2.0.
            M (float, optional): mass of the cart. Defaults to 8.0.
            l (float, optional): length of the pendulum. Defaults to 0.5.
            g (float, optional): gravity constant. Defaults to 9.8.
            dt (float, optional): simulation step. Defaults to 0.1.
            
        Returns:
            gym.Env: Inverted Pendulum environment
        """
        super(InvertedPendulumEnv, self).__init__()

        self.f = f
        self.f_noise = f_noise
        self.m = m
        self.M = M
        self.l = l
        self.g = g
        self.dt = dt

        # define observation space
        high = np.array([np.pi / 2., np.inf], dtype=np.float32)
        low = np.array([-np.pi / 2., -np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # define action space
        self.action_map = f * np.linspace(-1., 1., nA)
        self.action_space = spaces.Discrete(nA)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        th, thdot = self.state

        # get control
        u = self.action_map[action]
        u += self.f_noise * np.random.uniform(-1, 1)

        # get acceleration
        a = 1. / (self.m + self.M)
        thdotdot = self.g * math.sin(th) - a * self.m * self.l * math.pow(
            thdot, 2) * math.sin(2 * th) / 2 - a * math.cos(th) * u
        thdotdot /= 4 * self.l / 3 - a * self.m * self.l * math.pow(
            math.cos(th), 2)

        # update state
        thdot = thdot + thdotdot * self.dt
        th = th + thdot * self.dt

        self.state = np.array([th, thdot])

        # check if horizontal
        terminal = np.abs(th - 0.00001) >= (np.pi / 2)
        reward = -float(terminal)

        return self._get_obs(), reward, terminal, {}

    def reset(self):
        th = np.random.uniform(-5 * np.pi / 180, 5 * np.pi / 180)
        thdot = np.random.uniform(-5 * np.pi / 180, 5 * np.pi / 180)
        self.state = np.array([th, thdot])
        return self._get_obs()

    def _get_obs(self):
        th, thdot = self.state
        return np.array([th, thdot])
