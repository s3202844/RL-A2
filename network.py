from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam


def dqn(state_shape, n_actions, lr):
    """
    Get a Deep Q-learning Network, which inputs state vectors and outputs the Q-values for actions

    :param state_shape: The shape of state space
    :param n_actions: The number of actions
    :param lr: Learning rate
    :return: A DQN network
    """

    model = Sequential()
    model.add(Input(shape=state_shape))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(n_actions, activation='linear'))
    model.compile(loss=MSE, optimizer=Adam(learning_rate=lr))

    return model
