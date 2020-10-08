#!/usr/bin/python
import numpy as np
import problem5env as env


def iterative_policy_evaluation(V, S, theta=0.00001, gamma=0.9):
    # loop until the difference between v and V is smaller than delta
    while True:
        delta = 0
        # Iterate over all states
        for s in S:
            v = V[s]
            # Reset V[s] for recalculation
            state_value = 0
            for a in env.ACTION_LIST:
                s_prime, reward = env.step(s, a)
                # equiprobable random policy, transitions are deterministic.
                state_value += 0.25 * (reward + gamma * V[s_prime])
            # Update new value of state.
            V[s] = state_value
            # calculate delta to see if we have achieved v_pi
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return np.around(V, 1)

def main():
    V = np.zeros((5, 5), dtype=float)
    S = env.get_state_space()
    V = iterative_policy_evaluation(V, S)
    print(V)


if __name__ == '__main__':
    main()
