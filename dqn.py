import os
import gym
import copy
import argparse
import numpy as np

from agent import DQNAgent
from buffer import ReplayBuffer

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

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

    rewards = []
    max_reward = 0
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n
    agent = DQNAgent(state_shape, n_actions, lr, gamma, use_target)
    best_agent = copy.deepcopy(agent)
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
                replay_buffer.store(observation, action,
                                    reward, observation_next, done)
            else:
                state_batch.append(observation)
                action_batch.append(action)
                reward_batch.append(reward)
                state_next_batch.append(observation_next)
                done_batch.append(done)

            observation = observation_next
            # env.render()

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
        agent.update(state_batch, action_batch, reward_batch,
                     state_next_batch, done_batch)
        if e%100 == 0:
            print(f'Reward gained in epoch {e}: {reward_sum}')
        rewards += [reward_sum]
        if reward_sum > max_reward:
            max_reward = reward_sum
            best_agent = copy.deepcopy(agent)
        if max_reward == 500:
            break
    return best_agent, rewards


def evaluation(env, agent, epochs=20):
    rewards = []
    for _ in range(epochs):
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
        rewards += [reward_sum]
    return rewards


def experiment(epochs, lr, gamma, policy, epsilon, temp, use_replay, use_target,
               replay_buffer_size, replay_batch_size, target_update_epoch):
    env = gym.make("CartPole-v1")
    agent, rewards = dqn_training(env=env,
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
    print("Max reward gained during training: ", max(rewards))
    rewards += evaluation(env, agent)
    env.close()
    return rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for DQN.")
    parser.add_argument("-l", "--lr", type=float, default=0.1,
                        help="learning rate")
    parser.add_argument("-g", "--gamma", type=float, default=0.8,
                        help="discount")
    parser.add_argument("-s", "--strategy", type=str, default="egreedy",
                        help="Exploration strategy (default: egreedy)")
    parser.add_argument("-e", "--epsilon", type=float, default=0.4,
                        help="epsilon for e-greedy")
    parser.add_argument("-c", "--tempture", type=float, default=1.0,
                        help="tempture for softmax")
    parser.add_argument("-a", "--average", action="store_true",
                        help="Whether to repeat expeiment for average. (default: False)")
    parser.add_argument("-r", "--replay", action="store_true",
                        help="Whether to enable experience replay. (default: False)")
    parser.add_argument("-t", "--target_network", action="store_true",
                        help="Whether to enable target network. (default: False)")

    args = parser.parse_args()
    if args.replay:
        print("experience replay turned on")
    if args.target_network:
        print("target network turned on")

    epochs = 501
    lr = args.lr
    gamma = args.gamma

    policy = args.strategy
    epsilon = args.epsilon
    temp = args.tempture

    replay_buffer_size = 20000
    replay_batch_size = 100
    target_update_epoch = 3

    rewards = []
    times = 10 if args.average else 1
    for i in range(times):
        print(f"======================={i}=======================")
        rewards += [experiment(epochs, lr, gamma, policy, epsilon, temp,
                               args.replay, args.target_network,
                               replay_buffer_size, replay_batch_size,
                               target_update_epoch)]
    # path = "data/"+str(lr)+"_"+str(gamma)+"_"+policy + \
    #     "_"+str(epsilon)+"_"+str(temp)+"_"+str(replay_buffer_size) + \
    #     "_"+str(replay_batch_size)+"_"+str(target_update_epoch) + \
    #     "_"+str(args.replay)+"_"+str(args.target_network)
    # np.save(path+".npy", np.array(rewards))
