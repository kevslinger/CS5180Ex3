#!/usr/bin/python

import numpy as np
from scipy.stats import poisson


def create_reward_function():
    reward_table = np.zeros((21, 21, 11), dtype=np.float32)
    # Create the poisson random variables for the 2 locations.
    # note that we only need to know the R.V.s for the rental case
    # because we do not care about the next state for r(s, a)
    poisson_loc1 = poisson(3)
    poisson_loc2 = poisson(4)

    for x in range(len(reward_table)):
        for y in range(len(reward_table[x])):
            for z in range(len(reward_table[x][y])):
                reward = 0
                # True action space is [-5, ..., 0, ..., 5] not [0, ..., 10]
                a = z - 5
                # we cannot move more cars from y than we have in y, similar for x.
                if a > y or -a > x:
                    #print("x={}, y={}, a={}".format(x, y, a))
                    continue
                # at this point, we have the state (x, y) and the action a.
                # So, we know that we will start with a penalty of 2 * a,
                # because it costs $2 per car moved.
                reward = -2 * abs(a)
                # After that, we need to figure out the probability of each
                # number of cars being rented out are.
                # we multiply the probability times (10 * number of cars rented)
                # and then add that to the reward. We need to calculate this
                # for both location 1 and location 2.

                # remember that if a is positive, then cars are moved from loc2 to loc1
                # and if a is negative, then cars are moved from loc1 to loc2.
                total_cars_loc1 = x + a
                # we can rent out a total of total_cars_loc1.

                for i in range(total_cars_loc1 + 1):
                    reward += poisson_loc1.pmf(i) * (i * 10)

                total_cars_loc2 = y - a
                for i in range(total_cars_loc2 + 1):
                    reward += poisson_loc2.pmf(i) * (i * 10)

                reward_table[x, y, a] = reward
        #print("x={}".format(x))
    np.save('reward_function', reward_table)


if __name__ == '__main__':
    create_reward_function()
