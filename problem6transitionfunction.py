#!/usr/bin/python


from scipy.stats import poisson
import numpy as np
import pickle




def create_transition_function():
    transition_table = np.zeros((21, 21, 11, 21, 21), dtype=np.float32)
    lambda_rent_1 = 3
    lambda_return_1 = 3
    lambda_rent_2 = 4
    lambda_return_2 = 2
    poisson_rent_1 = poisson(lambda_rent_1)
    poisson_return_1 = poisson(lambda_return_1)
    poisson_rent_2 = poisson(lambda_rent_2)
    poisson_return_2 = poisson(lambda_return_2)

    for x in range(len(transition_table)):
        for y in range(len(transition_table[x])):
            for z in range(len(transition_table[x][y])):
                # We need to be sure we can actually perform the action.
                # If our state is (3, 2), then we cannot possible move 5 cars!
                # z is {0, ..., 10} so we subtract 5 to make it {-5, ., 0, ., 5}
                # e.g. if we have s=(9, 3) and z = 9, then 3 < (9 - 5) = 4
                # We just skip this iteration and move on!
                if y < (z - 5) or x < -(z - 5):
                    continue
                for w in range(len(transition_table[x][y][z])):
                    for v in range(len(transition_table[x][y][z][w])):
                        # z ranges from 0 to 10, but we want that to be -5 to 5
                        a = z - 5

                        prob_lot1 = 0
                        prob_lot2 = 0
                        # At this point, we have s = (x, y), a = z - 5, and s2 = (w, v)
                        # if a is 5, that means we moved 5 cars from location 2 to location 1
                        # diff is # cars in s2 minus the # of cars in previous state plus cars moved.
                        # If diff is positive, then we got more cars returned than rented out.
                        # if diff is negative, then we rented out more cars than we got returned.
                        lot1_diff = w - (x + a)
                        prob_lot1 = sum([poisson_rent_1.pmf(i) * poisson_return_1.pmf(i + lot1_diff) for i in range(x + a + 1)])
                        # for i in range(x + a + 1):
                        #     prob_lot1 += poisson_rent_1.pmf(i) * poisson_return_1.pmf(i+lot1_diff)
                        prob_lot1 += 1 - (poisson_rent_1.cdf(x) * poisson_return_1.cdf(x + lot1_diff))
                        # do the same thing for lot 2:
                        lot2_diff = v - (y - a)
                        # if lot2_diff is positive, then we gained lot2_diff returns more than rentals
                        # if lot2_diff is negative, then we rented out lot2_diff more than we got returned.
                        # if lot2_diff is 0, then we got the same number rented and returned.
                        prob_lot2 = sum([poisson_rent_2.pmf(i) * poisson_return_2.pmf(i + lot2_diff) for i in range(y - a + 1)])
                        # for i in range(y - a + 1):
                        #    prob_lot2 += poisson_rent_2.pmf(i) * poisson_return_2.pmf(i + lot2_diff)
                        # prob_lot2 += 1 - (poisson_rent_2.cdf(y) * poisson_return_2.cdf(y + lot2_diff))

                        transition_table[x, y, z, w, v] = prob_lot1 * prob_lot2
        print("x={}, y={}".format(x, y))
    np.save('transition_function2', transition_table)


# x = range(21), y = range(21), z = range(11), w = range(21), v = range(21)




if __name__ == '__main__':
    create_transition_function()

# p(s' | s, a) = Pr(S_t = s' | S_t-1 = s, A_t-1 = a} = \sum_{r\in R} p(s', r | s, a)
# r(s, a) = E[R_t | S_t-1 = s, A_t-1 = a] = \sum_{r \in R} r \sum_{s' \in S} p(s', r | s, a)


# p(s', r | s, a)
# if s is (10, 10) and a is 4
# Then, assuming no cars were rented or returned, we have (14, 6), and r = -8
# Suppose 2 cars were rented from each place.
# Then we have s' = (12, 4) and r = 32
# Lambda = 3 and 4 for rental requests, respecitvely
# and = 3 2 and returns, respectively.