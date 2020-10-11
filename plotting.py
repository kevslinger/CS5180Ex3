

import matplotlib.pyplot as plt
import numpy as np


# For each policy iteration, print out the policy we have at that iteration, just like in the book.
# also, print out the value function.
def main(paths, V_path):
    # we will use the same x-y grid for all plots to be consistent.
    x, y = np.meshgrid(np.arange(21), np.arange(21))
    i = 0
    for path in paths:
        # load in the policy.
        pi = np.load(path)

        fig, ax = plt.subplots()
        # Create a contour plot. Since all our values are exactly -5, -4, ..., 0, 1, ..., 5,
        # we create breaks at 0.9, 1.9,... to put them into the proper bins.
        cs = ax.contourf(x, y, pi, [-4.9, -3.9, -2.9, -1.9, -0.9, 0, 0.9, 1.9, 2.9, 3.9, 4.9, 5.9])

        ax.grid(c='k', ls='-', alpha=0.3)

        proxy = [plt.Rectangle((0, 0), 1, 1, fc = pc.get_facecolor()[0]) for
                 pc in cs.collections]
        # Use the same axis labels as the book uses.
        plt.xlabel('#Cars at second location')
        plt.ylabel('#Cars at first location')
        ax.set_title("Policy, π_{}".format(i))
        i += 1
        # Keep a legend. Without this, the color changes are sort of hard to decipher
        plt.legend(proxy, ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5", "6"])
    # We also want to plot the Value function at the end, just like the book.
    V = np.load(V_path)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # The book has this weird 3D graph going on, I found matplotlib's countour 3D looks similar.
    ax.contour3D(x, y, V, 100, cmap='binary')
    # Set labels just like book does.
    ax.set_xlabel("#cars at second location")
    ax.set_ylabel("#cars at first location")
    ax.set_title("V_pi4")
    plt.show()


if __name__ == '__main__':
    main(["pi_0.npy",
          "pi_1.npy",
          "pi_2.npy",
          "pi_3.npy",
          "pi_4.npy"],
         "V.npy")
