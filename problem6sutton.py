#!/usr/bin/python

import numpy as np
from scipy.stats import poisson
import tqdm


def load_p_and_r(P, R, lambda_requests, lambda_dropoffs):
    # Set up a requests and request probability variable.
    requests = 0
    request_prob = poisson(lambda_requests).pmf(requests)
    # Once we get to a probability below theta, it's so small we can ignore.
    while request_prob > theta:
        # for each possible number of starting cars (NOTE: we can have up to 25 (20 + send over 5)).
        for n in range(26):
            # Increase the reward by 10 * the probability of that reward * the number rented out
            R[n] += (10 * request_prob * min(requests, n))
        # Figure out how many cars were returned.
        dropoffs = 0
        drop_prob = poisson(lambda_dropoffs).pmf(dropoffs)
        # end the loop once our return probability is very small.
        while drop_prob > theta:
            # Remember we can have up to 25 cars.
            for n in range(26):
                satisfied_requests = min(requests, n)
                # can't have more than 20, or less than 0, cars at the end of the day.
                new_n = max(0, min(20, (n + dropoffs) - satisfied_requests))
                # In
                P[n][new_n] += request_prob * drop_prob
            dropoffs += 1
            drop_prob = poisson(lambda_dropoffs).pmf(dropoffs)
        requests += 1
        request_prob = poisson(lambda_requests).pmf(requests)


def backup_action(n1, n2, a):
    a = max(-n2, min(a, n1))
    a = max(-5, min(5, a))
    ret = -2 * abs(a)
    morning_n1 = int(n1 - a)
    morning_n2 = int(n2 + a)
    #if morning_n1 < 0 or morning_n2 < 0:
    #    return 0
    val = 0
    for new_n1 in range(21):
        for new_n2 in range(21):
            val += P1[morning_n1][new_n1] * P2[morning_n2][new_n2] * (R1[morning_n1] + R2[morning_n2] +
                                                                     gamma * V[new_n1][new_n2])
            #try:
            #    val += P1[morning_n1, new_n1] * P2[morning_n2, new_n2] * (R1[morning_n1] + R2[morning_n2] +
            #                                                            gamma * V[new_n1, new_n2])
            #except IndexError:
            #    print("morning_n1={}, new_n1={}, morning_n2={},new_n2={}".format(morning_n1, new_n1, morning_n2, new_n2))
            #    exit(0)
    return val


def policy_eval():
    delta = 1
    while theta < delta:
        outer_delta_list = []
        for n1 in range(21):
            inner_delta_list = []
            for n2 in range(21):
                #print("{}, {}".format(n1, n2))
                old_v = V[n1][n2]
                a = pi[n1][n2]
                V[n1][n2] = backup_action(n1, n2, a)
                inner_delta_list.append(abs(old_v - V[n1][n2]))
            outer_delta_list.append(max(inner_delta_list))
        delta = max(outer_delta_list)


def policy(n1, n2, epsilon=0.0000000001):
    best_value = -1
    best_action = None
    for a in range(max(-5, -n2), min(5, n2)+1):
        this_value = backup_action(n1, n2, a)
        if this_value > (best_value + epsilon):
            best_value = this_value
            best_action = a
    return best_action


def show_greedy_policy():
    for n1 in range(21):
        print()
        for n2 in range(21):
            print("{}".format(policy(n1, n2)))


def greedify():
    policy_improved = False
    for n1 in range(21):
        for n2 in range(21):
            b = pi[n1][n2]
            pi[n1][n2] = policy(n1, n2)
            if b != pi[n1][n2]:
                policy_improved = True
    show_policy()
    return policy_improved


def show_policy():
    for n1 in range(21):
        for n2 in range(21):
            print("{} ".format(pi[n1][n2]), end='')
        print()

def policy_iteration():
    count = 0
    #for i in tqdm.trange(100):
    while greedify():
        print("Inside greedify")
        policy_eval()
        count += 1
        print(count)


if __name__ == '__main__':
    V = np.zeros((21, 21), dtype=np.float32)
    lambda_requests1 = 3
    lambda_requests2 = 4
    lambda_dropoffs1 = 3
    lambda_dropoffs2 = 2
    pi = np.zeros((21, 21), dtype=int)
    P1 = np.zeros((26, 21))
    P2 = np.zeros((26, 21))
    R1 = np.zeros(26)
    R2 = np.zeros(26)
    gamma = 0.9
    theta = 0.0000001

    load_p_and_r(P1, R1, lambda_requests1, lambda_dropoffs1)
    load_p_and_r(P2, R2, lambda_requests2, lambda_dropoffs2)
    print("loaded p and r")

    policy_iteration()
    np.save("/home/kevin/Desktop/pi2", pi)
