import gym

if __name__ == '__main__':

    def get_action():
        i = input()
        match i:
            case "w":
                return 3
            case "d":
                return 2
            case "s":
                return 1
            case "a":
                return 0
            case _:
                print("wrong input")
                return get_action()

    env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=True)

    env.reset()
    env.render()

    gamma = 0.9
    step = 0
    tot_reward = 0
    done = False

    while not done:
        action = get_action()
        print(action)
        s, r, done, info = env.step(action)
        tot_reward += gamma ** step * r
        print(tot_reward)
        env.render()
        step += 1

    print(f"Your score = {tot_reward}")
