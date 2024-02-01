from nubo.test_functions import Hartmann6D
from algos.bm import bm
from algos.ei import ei
from algos.logei import logei
from algos.ucb import ucb


# parameters
func = Hartmann6D(minimise=False)           # objective function
noise = 0.0                                 # std of Gaussian noise added to func
bounds = func.bounds                        # upper and lower bounds for parameters
dynamic_dims = [5]                          # list of indices for dynamic dimensions
num_starts = 20                             # number of initial starts for the optimiser
num_samples = 100                           # number of samples for num_starts
evals = 100                                 # number of evaluations
runs = 30                                   # number of replicaions
stepsize = [0.05]                           # stepsize of random walk
num_tests = 25                              # number of test points
eval_stepsize = 10                          # increments of predictions
cores = 6
name = "hartmann"
start_str = "hartmann"
walk_str = "1d"


# # comparison
# bm(func, noise, bounds, dynamic_dims, num_starts, num_samples, evals, runs, stepsize, num_tests, eval_stepsize, cores, name, start_str, walk_str)
# print("Benchmark done.")

# ei(func, noise, bounds, dynamic_dims, num_starts, num_samples, evals, runs, stepsize, num_tests, eval_stepsize, cores, name, start_str, walk_str)
# print("Expected improvement done.")

# logei(func, noise, bounds, dynamic_dims, num_starts, num_samples, evals, runs, stepsize, num_tests, eval_stepsize, cores, name, start_str, walk_str)
# print("Log expected improvement done.")

# ucb(func, noise, bounds, dynamic_dims, num_starts, num_samples, evals, runs, stepsize, num_tests, eval_stepsize, cores, name, start_str, walk_str, 4)
# print("Upper confidence bound beta=4 done.")

# ucb(func, noise, bounds, dynamic_dims, num_starts, num_samples, evals, runs, stepsize, num_tests, eval_stepsize, cores, name, start_str, walk_str, 8)
# print("Upper confidence bound beta=8 done.")

# ucb(func, noise, bounds, dynamic_dims, num_starts, num_samples, evals, runs, stepsize, num_tests, eval_stepsize, cores, name, start_str, walk_str, 16)
# print("Upper confidence bound beta=16 done.")


# # noise
# ei(func, 0.025, bounds, dynamic_dims, num_starts, num_samples, evals, runs, stepsize, num_tests, eval_stepsize, cores, f"{name}_noisy_0.025", start_str, walk_str)
# print("Noise 0.025 done.")

# ei(func, 0.05, bounds, dynamic_dims, num_starts, num_samples, evals, runs, stepsize, num_tests, eval_stepsize, cores, f"{name}_noisy_0.05", start_str, walk_str)
# print("Noise 0.05 done.")

# ei(func, 0.1, bounds, dynamic_dims, num_starts, num_samples, evals, runs, stepsize, num_tests, eval_stepsize, cores, f"{name}_noisy_0.1", start_str, walk_str)
# print("Noise 0.1 done.")


# # multiple environmental conditions
# ei(func, noise, bounds, [3, 5], num_starts, num_samples, evals, runs, [0.10, 0.05], 50, eval_stepsize, cores, f"{name}_multi_2", start_str, "2d")
# print("2 environmental conditions done.")

# ei(func, noise, bounds, [0, 3, 5], num_starts, num_samples, evals, runs, [0.10, 0.10, 0.05], 75, eval_stepsize, cores, f"{name}_multi_3", start_str, "3d")
# print("3 environmental conditions done.")


# fluctuation
ei(func, noise, bounds, dynamic_dims, num_starts, num_samples, evals, runs, [0.1], num_tests, eval_stepsize, cores, f"{name}_stepsize_0.1", start_str, walk_str)
print("Fluctuation 0.1 done.")

ei(func, noise, bounds, dynamic_dims, num_starts, num_samples, evals, runs, [0.25], num_tests, eval_stepsize, cores, f"{name}_stepsize_0.25", start_str, walk_str)
print("Fluctuation 0.25 done.")

ei(func, noise, bounds, dynamic_dims, num_starts, num_samples, evals, runs, [0.5], num_tests, eval_stepsize, cores, f"{name}_stepsize_0.5", start_str, walk_str)
print("Fluctuation 0.5 done.")

ei(func, noise, bounds, dynamic_dims, num_starts, num_samples, evals, runs, [1.0], num_tests, eval_stepsize, cores, f"{name}_stepsize_1.0", start_str, walk_str)
print("Fluctuation 1.0 done.")


# parameter variability
ei(func, noise, bounds, [2], num_starts, num_samples, evals, runs, stepsize, num_tests, eval_stepsize, cores, f"{name}_variability_low", start_str, walk_str)
print("Low variability done.")

ei(func, noise, bounds, [0], num_starts, num_samples, evals, runs, stepsize, num_tests, eval_stepsize, cores, f"{name}_variability_high", start_str, walk_str)
print("High variability done.")
