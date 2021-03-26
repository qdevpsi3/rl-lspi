import numpy as np
from lspi.agents.agent import Agent
from sklearn.preprocessing import PolynomialFeatures


class PolynomialAgent(Agent):
    def __init__(self, env, degree, preprocess_obs=None):
        self.poly = PolynomialFeatures(degree)
        super(PolynomialAgent, self).__init__(env, preprocess_obs)

    def _get_features(self, obs):
        if not type(obs) in [np.ndarray, list, tuple]:
            obs = [obs]
        return self.poly.fit_transform([obs])[0]
