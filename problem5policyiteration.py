#!/usr/bin/python

import numpy as np
import problem5env as env
import math
import random


# Function for a policy, pi_v.
# Based on the value function,
# select the action which will give you
# the greatest return. Note: we also need gamma.
def policy(V, s, gamma):
    v = np.around(V, 1)
    action = ''
    max_value = -math.inf
    # Loop over action list to find the action with greatest value
    # If we have multiple, concatenate them.
    for a in env.ACTION_LIST:
        s_prime, r = env.step(s, a)
        value = r + gamma * v[s_prime]
        if value > max_value:
            action = a
            max_value = value
        elif value == max_value:
            action += '/' + a
    return action


# This function is used because we might have a tie
# where 2 actions are equally good. In this case,
# we break the tie arbitrarily.
def select_action_from_pi(pi, s):
    action = pi[s].decode('utf-8')
    if '/' in action:
        tokens = action.split('/')
        action = random.choice(tokens)
    return action


# Policy Iteration, as defined by the pseudocode in the book
def policy_iteration(S, V, pi, theta=0.001, gamma=0.9):
    # We want to keep on looping until we have a stable policy. That is,
    # The highest value action is the one selected by our policy for every state.
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
                # calculate delta to see if we have achieved v_pi
                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break
        # Perform policy improvement
        policy_stable = True
        # For every state. Do a 1-step lookahead and get the discounted return from
        # that state/action pair. Then, our policy will now pick the highest return.
        # If the previous policy had the same value as this new selection, then our
        # policy is stable for that state. Otherwise, it is not yet stable and we need
        # more iterations.
        for s in S:
            s_prime, reward = env.step(s, select_action_from_pi(pi, s))
            old_value = reward + gamma * V[s_prime]
            pi[s] = policy(V, s, gamma)
            s_prime, reward = env.step(s, select_action_from_pi(pi, s))
            new_value = reward + gamma * V[s_prime]
            if old_value != new_value:
                policy_stable = False
    return np.around(V, 1), pi


def main():
    # Create the state space
    S = env.get_state_space()
    V = np.zeros((5, 5), dtype=float)
    # Initialize a policy, pi, with a random action from the list
    # of possible actions, for each state.
    pi = np.chararray((5, 5), itemsize=20)
    for x in range(len(pi)):
        for y in range(len(pi[x])):
            pi[x, y] = random.choice(env.ACTION_LIST)
    V, pi = policy_iteration(S, V, pi)
    print(V)
    print(pi)


if __name__ == '__main__':
    main()
