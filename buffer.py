import random
from collections import deque


class ReplayBufferItem:
    """
    Define the items stored in the replay buffer
    """

    def __init__(self, state, action, reward, state_next, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.state_next = state_next
        self.done = done


class ReplayBuffer:
    """
    A buffer for storing replay records
    """

    def __init__(self, size):
        self.buffer_queue = deque(maxlen=size)

    def store(self, state, action, reward, state_next, done):
        item = ReplayBufferItem(state, action, reward, state_next, done)
        self.buffer_queue.append(item)

    def get_batch(self, batch_size):
        sample_num = min(batch_size, len(self.buffer_queue))
        return random.sample(self.buffer_queue, sample_num)
