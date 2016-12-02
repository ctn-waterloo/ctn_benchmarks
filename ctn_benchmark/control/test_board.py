'''
Ben Morcos
2016-11-28

Iterate over adaptive_bias.py with different seeds
'''
import ctn_benchmark as ct

N = [100, 200, 500, 1000, 2000, 5000, 10000]  # test values
D = [1, 3, 6]  # input dim = output dim
n_trials = 50  # run through all N and D with different seed

for i in range(n_trials):  # seed
    for j, n in enumerate(N):  # Number of neurons
        for k, d in enumerate(D):  # number of dimensions
            print("\nseed: " + str(i) + ";  N: " + str(n) + ";  D: " + str(d))
            ct.AdaptiveBias().run(adapt=True, noise=0, delay=0, filter=0,
                                  n_neurons=n, T=20, seed=i, D=d,
                                  learning_rate=0.0001)
