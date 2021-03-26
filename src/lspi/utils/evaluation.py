def evaluate_policy(agent, env, max_length=1000, n_eval_episodes=10):
    """Runs policy for ``n_eval_episodes`` episodes.
    
    Adapted from :
    https://stable-baselines.readthedocs.io/en/master/_modules/stable_baselines/common/evaluation.html

    Args:
        agent (lspi.agents.Agent): features policy.
        env (gym.Env): gym environment.
        max_length (int, optional): maximum episode length. Defaults to 1000.
        n_eval_episodes (int, optional): number of episode to evaluate the agent. Defaults to 10.

    Returns:
        episode_rewards (List[float]): list of reward per episode
        episode_lengths (List[int]): list of length per episode
    """
    episode_rewards, episode_lengths = [], []
    while len(episode_rewards) < n_eval_episodes:
        obs = env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0
        while not (done or episode_length == max_length):
            action = agent.predict(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    return episode_rewards, episode_lengths
