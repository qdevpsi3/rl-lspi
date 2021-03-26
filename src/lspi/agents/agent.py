import numpy as np


class Agent:
    def __init__(self, env, preprocess_obs=None):
        if preprocess_obs is None:
            preprocess_obs = lambda x: x

        self.env = env
        self.action_size = self.env.action_space.n
        self.preprocess_obs = preprocess_obs
        self.features_size = self.get_features_size()
        self.init_weights()

    def init_weights(self, scale=1.):
        size = self.features_size * self.action_size
        self.weights = np.random.normal(size=size, scale=scale)

    def set_weights(self, weights):
        self.weights = weights

    def get_features_size(self):
        obs = self.env.observation_space.sample()
        features = self.get_features(obs)
        return len(features)

    def get_features(self, obs):
        obs = self.preprocess_obs(obs)
        return self._get_features(obs)

    def _get_features(self, obs):
        pass

    def predict(self, obs):
        values = np.dot(
            self.weights.reshape(self.action_size, self.features_size),
            self.get_features(obs))
        action = np.argmax(values)
        return action
