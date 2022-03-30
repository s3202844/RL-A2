import numpy as np

from network import dqn


class DQNAgent:
    """
    An agent using DQN to predict Q-values

    Args:
        state_shape: The shape of state space
        n_actions: The number of actions
        lr: Learning rate
        gamma: The discount factor for estimating future Q-values
        use_target: If true, then use target network to predict Q-values
    """

    def __init__(self,
                 state_shape,
                 n_actions,
                 lr=0.1,
                 gamma=1.0,
                 use_target=False):

        self.state_shape = state_shape
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma

        self.use_target = use_target
        self.net = dqn(state_shape, n_actions, lr)
        self.target_net = dqn(state_shape, n_actions, lr)

    def select_action(self, state, policy=None, epsilon=None, temp=None):
        state = np.expand_dims(state, axis=0)
        if self.use_target:
            q_values = self.target_net.predict(state)
        else:
            q_values = self.net.predict(state)
        q_values = q_values[0]

        if policy == 'egreedy':
            if epsilon is None:
                raise KeyError("Provide an epsilon")

            if np.random.rand() < epsilon:
                action = np.random.randint(0, self.n_actions)
            else:
                action = np.argmax(q_values)

        elif policy == 'softmax':
            if temp is None:
                raise KeyError("Provide a temperature")

            x = q_values / temp
            z = x - max(x)
            p = np.exp(z) / np.sum(np.exp(z))
            action = np.random.choice(np.arange(0, self.n_actions), p=p)

        else:
            action = np.argmax(q_values)

        return action

    def update(self, state_batch, action_batch, reward_batch, state_next_batch, done_batch):
        if self.use_target:
            q_value_batch = self.target_net.predict(state_batch)
            q_value_next_batch = self.target_net.predict(state_next_batch)
        else:
            q_value_batch = self.net.predict(state_batch)
            q_value_next_batch = self.net.predict(state_next_batch)

        q_value_next_batch = q_value_next_batch.max(axis=1)
        for i in range(len(state_batch)):
            a = action_batch[i]
            r = reward_batch[i]
            q_next = q_value_next_batch[i]
            done = done_batch[i]

            q_new = r
            if not done:
                q_new += self.gamma * q_next
            q_value_batch[i][a] = q_new

        self.net.fit(state_batch, q_value_batch, verbose=0)

    def update_target(self):
        self.target_net.set_weights(self.net.get_weights())
