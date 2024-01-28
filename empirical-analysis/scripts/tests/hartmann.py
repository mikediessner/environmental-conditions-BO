from nubo.test_functions import Hartmann6D


# parameters
func = Hartmann6D(minimise=False)           # objective function
noise = 0.0                                 # std of Gaussian noise added to func
bounds = func.bounds                        # upper and lower bounds for parameters
dynamic_dims = [5]                          # list of indices for dynamic dimensions
num_starts = 20                             # number of initial starts for the optimiser
num_samples = 500                           # number of samples for num_starts
evals = 100                                 # number of evaluations
runs = 30                                   # number of replicaions
stepsize = [0.05]                           # stepsize of random walk
num_tests = 25                              # number of test points
eval_stepsize = 10                          # increments of predictions
cores = 6
name = "hartmann"
start_str = "hartmann"
walk_str = "1d"

from algos.bm import bm
bm(func, noise, bounds, dynamic_dims, num_starts, num_samples, evals, runs, stepsize, num_tests, eval_stepsize, cores, name, start_str, walk_str)
print("Benchmark done.")

from algos.ei import ei
ei(func, noise, bounds, dynamic_dims, num_starts, num_samples, evals, runs, stepsize, num_tests, eval_stepsize, cores, name, start_str, walk_str)
print("Expected improvement done.")

from algos.logei import logei
logei(func, noise, bounds, dynamic_dims, num_starts, num_samples, evals, runs, stepsize, num_tests, eval_stepsize, cores, name, start_str, walk_str)
print("Log expected improvement done.")

from algos.ucb import ucb
ucb(func, noise, bounds, dynamic_dims, num_starts, num_samples, evals, runs, stepsize, num_tests, eval_stepsize, cores, name, start_str, walk_str, 4)
print("Upper confidence bound done.")

from algos.ucb import ucb
ucb(func, noise, bounds, dynamic_dims, num_starts, num_samples, evals, runs, stepsize, num_tests, eval_stepsize, cores, name, start_str, walk_str, 8)
print("Upper confidence bound done.")

from algos.ucb import ucb
ucb(func, noise, bounds, dynamic_dims, num_starts, num_samples, evals, runs, stepsize, num_tests, eval_stepsize, cores, name, start_str, walk_str, 16)
print("Upper confidence bound done.")
