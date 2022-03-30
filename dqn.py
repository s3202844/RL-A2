import gym
import argparse
import numpy as np

from agent import DQNAgent
from buffer import ReplayBuffer


def dqn_training(env,
                 epochs=100,
                 lr=0.1,
                 gamma=1.0,
                 policy=None,
                 epsilon=None,
                 temp=None,
                 use_replay=False,
                 use_target=False,
                 replay_buffer_size=1000,
                 replay_batch_size=10,
                 target_update_interval=5):
    """
    Train a DQN agent in a given environment

    :param env: The environment the agent interact with
    :param epochs: Number of epochs for training
    :param lr: Learning rate
    :param gamma: The discount factor
    :param policy: The action selection policy
    :param epsilon: Parameter for 'e-greedy' policy
    :param temp: Temperature for 'softmax' policy
    :param use_replay: If true, then apply replay buffer during training
    :param use_target: If true, then apply target network during training
    :param replay_buffer_size: The size of replay buffer
    :param replay_batch_size: The number of records got from the replay buffer per epoch
    :param target_update_interval: The interval of epoch for updating the target network
    :return A trained agent
    """

    state_shape = env.observation_space.shape
    n_actions = env.action_space.n
    agent = DQNAgent(state_shape, n_actions, lr, gamma, use_target)
    replay_buffer = ReplayBuffer(replay_buffer_size)

    for e in range(epochs):
        state_batch = []
        action_batch = []
        reward_batch = []
        state_next_batch = []
        done_batch = []

        if use_target and e % target_update_interval == 0:
            agent.update_target()

        observation = env.reset()
        reward_sum = 0
        done = False
        while not done:
            action = agent.select_action(observation, policy, epsilon, temp)
            observation_next, reward, done, _ = env.step(action)
            reward_sum += reward

            if use_replay:
                replay_buffer.store(observation, action, reward, observation_next, done)
            else:
                state_batch.append(observation)
                action_batch.append(action)
                reward_batch.append(reward)
                state_next_batch.append(observation_next)
                done_batch.append(done)

            observation = observation_next
            env.render()

        if use_replay:
            batch = replay_buffer.get_batch(replay_batch_size)
            for b in batch:
                state_batch.append(b.state)
                action_batch.append(b.action)
                reward_batch.append(b.reward)
                state_next_batch.append(b.state_next)
                done_batch.append(b.done)

        state_batch = np.array(state_batch)
        state_next_batch = np.array(state_next_batch)
        agent.update(state_batch, action_batch, reward_batch, state_next_batch, done_batch)
        print(f'Reward gained in epoch {e}: {reward_sum}')
    return agent


def evaluation(env, agent):
    observation = env.reset()
    reward_sum = 0
    done = False
    while not done:
        action = agent.select_action(observation)
        observation_next, reward, done, _ = env.step(action)
        reward_sum += reward
        observation = observation_next
        env.render()
    print(f'Reward gained by the agent:{reward_sum}')


def experiment(use_replay, use_target):
    epochs = 500
    lr = 0.1
    gamma = 0.8

    policy = 'egreedy'
    epsilon = 0.4
    temp = 1.0

    replay_buffer_size = 2000
    replay_batch_size = 20
    target_update_epoch = 3

    env = gym.make("CartPole-v1")
    agent = dqn_training(env=env,
                         epochs=epochs,
                         lr=lr,
                         gamma=gamma,
                         policy=policy,
                         epsilon=epsilon,
                         temp=temp,
                         use_replay=use_replay,
                         use_target=use_target,
                         replay_buffer_size=replay_buffer_size,
                         replay_batch_size=replay_batch_size,
                         target_update_interval=target_update_epoch)
    evaluation(env, agent)
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for DQN.")
    parser.add_argument("-s", "--strategy", type=str, default="e-greedy",
                        help="Exploration strategy (default: e-greedy)")
    parser.add_argument("-r", "--replay", action="store_true",
                        help="Whether to enable experience replay (default: False)")
    parser.add_argument("-t", "--target_network", action="store_true",
                        help="Whether to enable target network (default: False)")

    args = parser.parse_args()
    if args.replay:
        print("experience replay turned on")
    if args.target_network:
        print("target network turned on")

    experiment(args.replay, args.target_network)
