import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter


def cut(rewards, head=True):
    res = []
    for i in range(rewards.shape[0]):
        if head:
            res += [rewards[i][:-20]]
        else:
            res += [rewards[i][-20:]]
    return np.array(res)


def curve(path, head=True):
    data = cut(np.load(path), head)
    data = np.mean(data, axis=0)
    window_length = 51 if head else 3
    data = savgol_filter(data, window_length, 1)
    return data

def df_curve(path, head=True):
    smoothed = []
    data = cut(np.load(path), head)
    window_length = 51 if head else 3
    for i in range(data.shape[0]):
        smoothed += [savgol_filter(data[i], window_length, 1)]
    smoothed = np.array(smoothed)
    df_data = pd.DataFrame(smoothed)
    return df_data


name = [
    "0.1_0.8_egreedy_0.2_1.0_20000_100_15_False_False.npy",
    "0.2_0.8_egreedy_0.2_1.0_20000_100_15_False_False.npy",
    "0.4_0.8_egreedy_0.2_1.0_20000_100_15_False_False.npy",
    "0.1_1.0_egreedy_0.2_1.0_20000_100_15_False_False.npy",
    "0.1_0.8_egreedy_0.05_1.0_20000_100_15_False_False.npy",
    "0.1_0.8_egreedy_0.4_1.0_20000_100_15_False_False.npy",
    "0.1_0.8_softmax_0.2_0.1_20000_100_15_False_False.npy",
    "0.1_0.8_softmax_0.2_1.0_20000_100_15_False_False.npy",
    "0.1_0.8_softmax_0.2_10.0_20000_100_15_False_False.npy",
    "0.1_0.8_egreedy_0.2_1.0_20000_100_15_True_False.npy",
    "0.1_0.8_egreedy_0.2_1.0_20000_100_15_False_True.npy",
    "0.1_0.8_egreedy_0.2_1.0_20000_100_15_True_True.npy",
]

for i in range(len(name)):
    name[i] = "data/" + name[i]

# learning rates
rewards = curve(name[0])
plt.plot(rewards, label="$\\alpha$=0.1")
rewards = curve(name[1])
plt.plot(rewards, label="$\\alpha$=0.2")
rewards = curve(name[2])
plt.plot(rewards, label="$\\alpha$=0.4")
plt.xlabel("epoch")
plt.ylabel("reward (Avg. over 10 repetitions)")
plt.title("Influence of learning rates on DQN.")
plt.legend()
plt.savefig("results/lr.png")
plt.clf()

# gamma
rewards = curve(name[0])
plt.plot(rewards, label="$\gamma$=0.8")
rewards = curve(name[3])
plt.plot(rewards, label="$\gamma$=1")
plt.xlabel("epoch")
plt.ylabel("reward (Avg. over 10 repetitions)")
plt.title("Influence of $\gamma$ on DQN.")
plt.legend()
plt.savefig("results/gamma.png")
plt.clf()

# epsilon
rewards = curve(name[4])
plt.plot(rewards, label="$\epsilon$=0.05")
rewards = curve(name[0])
plt.plot(rewards, label="$\epsilon$=0.2")
rewards = curve(name[5])
plt.plot(rewards, label="$\epsilon$=0.4")
plt.xlabel("epoch")
plt.ylabel("reward (Avg. over 10 repetitions)")
plt.title("Influence of $\epsilon$ of $\epsilon$-greedy on DQN.")
plt.legend()
plt.savefig("results/epsilon.png")
plt.clf()

# tempture
rewards = curve(name[6])
plt.plot(rewards, label="$\\tau$=0.1")
rewards = curve(name[7])
plt.plot(rewards, label="$\\tau$=1.0")
rewards = curve(name[8])
plt.plot(rewards, label="$\\tau$=10.0")
plt.xlabel("epoch")
plt.ylabel("reward (Avg. over 10 repetitions)")
plt.title("Influence of $\\tau$ of softmax on DQN.")
plt.legend()
plt.savefig("results/tempture.png")
plt.clf()

# strategies
rewards = curve(name[0])
plt.plot(rewards, label="$\epsilon$-greedy")
rewards = curve(name[7])
plt.plot(rewards, label="softmax")
plt.xlabel("epoch")
plt.ylabel("reward (Avg. over 10 repetitions)")
plt.title("Influence of strategies on DQN.")
plt.legend()
plt.savefig("results/strategies.png")
plt.clf()

# +ER
rewards = df_curve(name[0])
y = rewards.mean(axis=0)
plt.plot(y[:200], label="DQN")
rewards = df_curve(name[9])
y = rewards.mean(axis=0)
plt.plot(y, label="DQN+ER")
plt.xlabel("epoch")
plt.ylabel("reward (Avg. over 10 repetitions)")
plt.title("DQN vs DQN+ER")
plt.legend()
plt.savefig("results/ER.png")
plt.clf()


# +TN
rewards = df_curve(name[0])
mstd = rewards.std(axis=0)
y = rewards.mean(axis=0)
plt.plot(y[:200], label="DQN")
plt.fill_between(list(range(200)), (y-mstd)[:200], (y+mstd)[:200], alpha=0.2)

rewards = df_curve(name[10])
mstd = rewards.std(axis=0)
y = rewards.mean(axis=0)
plt.plot(y, label="DQN+TN")
plt.fill_between(list(range(rewards.shape[1])), y-mstd, y+mstd, alpha=0.2)
plt.xlabel("epoch")
plt.ylabel("reward (Avg. over 10 repetitions)")
plt.title("DQN vs DQN+TN")
plt.legend()
plt.savefig("results/TN.png")
plt.clf()

# +TN
rewards = df_curve(name[0])
mstd = rewards.std(axis=0)
y = rewards.mean(axis=0)
plt.plot(y[:200], label="DQN")
# plt.fill_between(list(range(200)), (y-mstd)[:200], (y+mstd)[:200], alpha=0.2)

rewards = df_curve(name[11])
mstd = rewards.std(axis=0)
y = rewards.mean(axis=0)
plt.plot(y, label="DQN+ER+TN")
# plt.fill_between(list(range(rewards.shape[1])), y-mstd, y+mstd, alpha=0.2)
plt.xlabel("epoch")
plt.ylabel("reward (Avg. over 10 repetitions)")
plt.title("DQN vs DQN+ER+TN")
plt.legend()
plt.savefig("results/ER_TN.png")
plt.clf()

