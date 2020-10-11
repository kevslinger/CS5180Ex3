

import matplotlib.pyplot as plt
import numpy as np


def main(paths):
    for path in paths:
        pi = np.load(path)
        #print(np.shape(pi))
        x, y = np.meshgrid(np.arange(21), np.arange(21))
        #y = [[i] * 21 for i in range(21)]

        fig, ax = plt.subplots()
        cs = ax.contourf(x, y, pi, [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
        ax.set_title("Policy, pi")

        ax.grid(c='k', ls='-', alpha=0.3)

        proxy = [plt.Rectangle((0, 0), 1, 1, fc = pc.get_facecolor()[0]) for
                 pc in cs.collections]

        plt.xlabel('#Cars at second location')
        plt.ylabel('#Cars at first location')
        plt.legend(proxy, ["-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"])

    plt.show()

if __name__ == '__main__':
   # main(["/home/kevin/Desktop/pi_0_2.npy",
   #       "/home/kevin/Desktop/pi_1_2.npy",
   #       "/home/kevin/Desktop/pi_2_2.npy",
   #       "/home/kevin/Desktop/pi_3_2.npy",
   #       "/home/kevin/Desktop/pi_4_2.npy"])
   main(["/home/kevin/Desktop/sutton.npy"])