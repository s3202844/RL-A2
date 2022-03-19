import gym
import argparse


def example4gym():
    env = gym.make("CartPole-v1")
    env.reset()
    print("The number of agent's actions:", env.action_space.n)
    for _ in range(100):
        action = env.action_space.sample()
        _, reward, done, _ = env.step(action)
        if done:
            env.reset()
        env.render()
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

    example4gym()
