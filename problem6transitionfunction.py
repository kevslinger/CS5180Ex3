#!/usr/bin/python


from scipy.stats import poisson
import numpy as np




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

    for x in range(transition_table):
        for y in range(transition_table[x]):
            for z in range(transition_table[x][y]):
                for w in range(transition_table[x][y][z]):
                    for a in range(transition_table[x][y][z][w]):
                        prob_lot1 = 0
                        # At this point, we have s = (x, y), a = z, and s2 = (w, a)
                        # if a is 5, that means we moved 5 cars from location 2 to location 1
                        # diff is # cars in s2 minus the # of cars in previous state plus cars moved.
                        # If diff is positive, then we got more cars returned than rented out.
                        # if diff is negative, then we rented out more cars than we got returned.
                        lot1_diff = w - (x + z)
                        for i in range(x+z+1):
                            prob_lot1 += poisson_rent_1.pmf(i) * poisson_return_1.pmf(i+lot1_diff)
                        prob_lot1 += 1 - (poisson_rent_1.cdf(x) * poisson_return_1.cdf(x + lot1_diff))
                        # do the same thing for lot 2:
                        prob_lot2 = 0
                        lot2_diff = a - (y - z)
                        # if lot2_diff is positive, then we gained lot2_diff returns more than rentals
                        # if lot2_diff is negative, then we rented out lot2_diff more than we got returned.
                        # if lot2_diff is 0, then we got the same number rented and returned.
                        for i in range(y - z + 1):
                            prob_lot2 += poisson_rent_2.pmf(i) * poisson_return_2.pmf(i + lot2_diff)
                        prob_lot2 += 1 - (poisson_rent_2.cdf(y) * poisson_return_2.cdf(y + lot2_diff))

                        transition_table[x, y, z, w, a] = prob_lot1 * prob_lot2






# Function to save the transition function of Jack's car rental environment
def open_to_close():
    #
    #transition_table = np.zeros((21, 21), dtype=np.float32)

    # Give the probability that we end the day with x cars, given we started with y cars.
    location1_prob_table = np.zeros((21, 26), dtype=np.float32)
    rental_lambda = 3
    return_lambda = 3
    rental_rv = poisson(3)
    return_rv = poisson(3)
    for x in range(len(location1_prob_table)):
        for y in range(len(location1_prob_table[x])):
            # The probability of having x cars available at the end of the day, given we started with y
            # This is equal to the sum of probabilities that we rent out some number of cars and get others returned
            # suppose we start with 15 and end with 10. The prob of that is
            # the sum of probabilities that: we rent out 5 and get 0 returned.
            #                               we rent out 6 and get 1 returned.
            #                                  .... up to renting out 15 and getting 10 returned.
            location1_prob_table[x, y] = sum([lambda x2, y2: rental_rv.pmf(x2) * return_rv.pmf(y2), range(y - x, y), range(x)])


# p(s' | s, a) = Pr(S_t = s' | S_t-1 = s, A_t-1 = a} = \sum_{r\in R} p(s', r | s, a)
# r(s, a) = E[R_t | S_t-1 = s, A_t-1 = a] = \sum_{r \in R} r \sum_{s' \in S} p(s', r | s, a)


# p(s', r | s, a)
# if s is (10, 10) and a is 4
# Then, assuming no cars were rented or returned, we have (14, 6), and r = -8
# Suppose 2 cars were rented from each place.
# Then we have s' = (12, 4) and r = 32
# Lambda = 3 and 4 for rental requests, respecitvely
# and = 3 2 and returns, respectively.