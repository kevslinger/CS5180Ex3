#!/usr/bin/python

from scipy.stats import poisson

GAMMA = 0.9

def poiss(lamb, n):
    return poisson()

# computes the probability of ending the day with s_end \in [0, 20] cars
# given that the location started the day with s_start \in [0, 20+5] cars
# will also compute the average reward the location experiences during the dat
# given the location started the day with s_start cars.
# can be pre-computed for all 26 possible starting numbers of cars for each location.
# Then, computer the joint dynamics between the 2 locations by considering the
# deterministic overnight dynamics, and then combine the appropriate "open to close"
# dynamics for each location.
def open_to_close():
    pass


def main():
    pass


if __name__ == '__main__':
    main()


# p(s' | s, a) = Pr(S_t = s' | S_t-1 = s, A_t-1 = a} = \sum_{r\in R} p(s', r | s, a)
# r(s, a) = E[R_t | S_t-1 = s, A_t-1 = a] = \sum_{r \in R} r \sum_{s' \in S} p(s', r | s, a)


#s is the number of cars at each location at the end of the day
# time steps are days
# actions are the net number of cars moved between the two locations overnight
# state = (10, 8) meaning there are 10 cars in location 1 and 8 cars in location 2
# action = 4 means we are moving 4 cars from location 2 to location 1
#   (so location 2 loses 4 cars and location 1 gains 4 cars).
# action state is [-5, ..., 5]
# state space is (0,0) --> (20, 20)?
# then the next morning we would have
# for handling what happens between action and state,we need the open to close function
# although we have (14, 4) in the morning, we will rent out cars and also get cars returned to us.
# with some probabilities, each.
# our total reward is $10 * cars rented - 2 * cars moved.
# maximum of 5 cars can be moved at one night.





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
def policy_iteration(S, V, pi, theta=0.1, gamma=0.9):
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


