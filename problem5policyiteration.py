#!/usr/bin/python

import numpy as np
import problem5env as env
import math
import random


def policy(V, s, gamma):
    v = np.around(V, 1)
    action = ''
    max_value = -math.inf
    for a in env.ACTION_LIST:
        s_prime, r = env.step(s, a)
        value = r + gamma * v[s_prime]
        #print("Value: {}, Max Value: {}".format(value, max_value))
        if value > max_value:
            action = a
            max_value = value
        elif value == max_value:
            action += '/' + a
    #print(action)
    return action


def select_action_from_pi(pi, s):
    action = pi[s].decode('utf-8')
    if '/' in action:
        tokens = action.split('/')
        action = random.choice(tokens)
    return action


def policy_iteration(S, V, pi, theta=0.001, gamma=0.9):

    policy_stable = False
    while not policy_stable:
        # Perform policy evaluation
        # loop until the difference between v and V is smaller than delta
        while True:
            delta = 0
            # Iterate over all states
            for s in S:
                v = V[s]
                # Reset V[s] for recalculation
                state_value = 0
                s_prime, reward = env.step(s, select_action_from_pi(pi, s))
                # We start with an arbitrarily initialized policy, then improve.
                # the transitions are deterministic.
                state_value += reward + gamma * V[s_prime]
                # Update new value of state.
                V[s] = state_value
                #print("v = {} and V[s] = {}".format(v, V[s]))
                # calculate delta to see if we have achieved v_pi
                delta = max(delta, abs(v - V[s]))
                #print(delta)
            #print(V)
            if delta < theta:
                break
        print("Got to policy improvement part")
        print(np.around(V, 1))
        # Perform policy improvement
        policy_stable = True
        for s in S:
            s_prime, reward = env.step(s, select_action_from_pi(pi, s))
            old_value = reward + gamma * V[s_prime]
            pi[s] = policy(V, s, gamma)
            s_prime, reward = env.step(s, select_action_from_pi(pi, s))
            new_value = reward + gamma * V[s_prime]
            if old_value != new_value:
                policy_stable = False

    return np.around(V, 1), pi

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
    S = env.get_state_space()
    V = np.zeros((5, 5), dtype=float)
    pi = np.chararray((5, 5), itemsize=20)
    for x in range(len(pi)):
        for y in range(len(pi[x])):
            pi[x, y] = random.choice(env.ACTION_LIST)
    V, pi = policy_iteration(S, V, pi)
    print(V)
    print(pi)


if __name__ == '__main__':
    main()
