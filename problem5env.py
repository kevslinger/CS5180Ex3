#!/usr/bin/python
import random


NORTH = 'up'
SOUTH = 'down'
EAST = 'right'
WEST = 'left'
ACTION_LIST = [NORTH, SOUTH, EAST, WEST]


# Inputs:
#   state: the current x, y position of the agent
#   action: the direction the agent wants to move
# Outputs:
#   s_prime: the new state the agent is in.
#   reward: the scalar reward for the agent. either 0, -1, 10, or 5
def step(state, action):
    # If we're in one of the goal states, all actions receive the same reward
    # and result in the same next state.
    # This is the A state. The indexing is strange because of numpy's array indexing.
    if state == (0, 1):
        reward = 10
        s_prime = (4, 1)
    # This is the B state.
    elif state == (0, 3):
        reward = 5
        s_prime = (2, 3)
    # Try to move north
    # The directions might not look intuitive, but again, this is because numpy's indexing and
    # our intuitive indexing are not the same.
    elif action == NORTH:
        # This would cause us to fall off, so we instead get a reward -1 and don't move
        if state[0] - 1 < 0:
            reward = -1
            s_prime = state
        else:
            reward = 0
            s_prime = (state[0] - 1, state[1])
    # Try to move south
    elif action == SOUTH:
        # prevent moving off the edge
        if state[0] + 1 > 4:
            reward = -1
            s_prime = state
        else:
            reward = 0
            s_prime = (state[0] + 1, state[1])
    # Try to move east
    elif action == EAST:
        # Prevent moving off the edge
        if state[1] + 1 > 4:
            reward = -1
            s_prime = state
        else:
            reward = 0
            s_prime = (state[0], state[1] + 1)
    # Finally, try to move west
    elif action == WEST:
        if state[1] - 1 < 0:
            reward = -1
            s_prime = state
        else:
            reward = 0
            s_prime = (state[0], state[1] - 1)
    return s_prime, reward


def get_state_space():
    S = []
    # Create state space
    for x in range(5):
        for y in range(5):
            S.append((x, y))
    return S
