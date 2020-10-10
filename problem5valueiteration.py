#!/usr/bin/python
import numpy as np
import problem5env as env
import math


def value_iteration(S, theta=0.001, gamma=0.9):
    V = np.zeros((5, 5), dtype=float)
    while True:
        delta = 0
        for s in S:
            v = V[s]
            # keep track of all the values in this state
            value_list = []
            for a in env.ACTION_LIST:
                s_prime, r = env.step(s, a)
                value_list.append(r + gamma * V[s_prime])
            V[s] = max(value_list)
            delta = max(delta, abs(v - V[s]))
            #print(delta)
        if delta < theta:
            break
    # Round to 1 decimal just like the book.
    V = np.around(V, 1)
    def policy(s):
        action = ''
        max_value = - math.inf
        for a in env.ACTION_LIST:
            s_prime, r = env.step(s, a)
            value = r + gamma * V[s_prime]
            if value > max_value:
                max_value = value
                action = a
            elif value == max_value:
                action += '/' + a
        return action

    return V, policy


def main():
    S = env.get_state_space()
    V, policy = value_iteration(S)
    print(np.around(V, 1))
    optimal_action = [[''] * 5] * 5
    for x in range(len(optimal_action)):
        for y in range(len(optimal_action[x])):
            optimal_action[x][y] = policy((x, y))
        print(optimal_action[x])


if __name__ == '__main__':
    main()