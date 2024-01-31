import torch
from nubo.test_functions import Levy
from algos.bm import bm
from algos.ei import ei
from algos.logei import logei
from algos.ucb import ucb


# parameters
func = Levy(2, minimise=True)               # objective function
noise = 0.0                                 # std of Gaussian noise added to func
bounds = torch.tensor([[-7.5, -10.], 
                       [ 7.5,  10.]])       # upper and lower bounds for parameters
dynamic_dims = [1]                          # list of indices for dynamic dimensions
num_starts = 20                             # number of initial starts for the optimiser
num_samples = 100                           # number of samples for num_starts
evals = 100                                 # number of evaluations
runs = 30                                   # number of replicaions
stepsize = [1.5]                            # stepsize of random walk
num_tests = 25                              # number of test points
eval_stepsize = 10                          # increments of predictions
cores = 6
name = "levy"
start_str = "levy"
walk_str = "1d"


# comparison
bm(func, noise, bounds, dynamic_dims, num_starts, num_samples, evals, runs, stepsize, num_tests, eval_stepsize, cores, f"{name}_benchmark", start_str, walk_str)
print("Benchmark done.")

ei(func, noise, bounds, dynamic_dims, num_starts, num_samples, evals, runs, stepsize, num_tests, eval_stepsize, cores, f"{name}_ei", start_str, walk_str)
print("Expected improvement done.")

logei(func, noise, bounds, dynamic_dims, num_starts, num_samples, evals, runs, stepsize, num_tests, eval_stepsize, cores, f"{name}_ucb4", start_str, walk_str)
print("Log expected improvement done.")

ucb(func, noise, bounds, dynamic_dims, num_starts, num_samples, evals, runs, stepsize, num_tests, eval_stepsize, cores, f"{name}_ucb8", start_str, walk_str, 4)
print("Upper confidence bound done.")

ucb(func, noise, bounds, dynamic_dims, num_starts, num_samples, evals, runs, stepsize, num_tests, eval_stepsize, cores, f"{name}_logei", start_str, walk_str, 8)
print("Upper confidence bound done.")

ucb(func, noise, bounds, dynamic_dims, num_starts, num_samples, evals, runs, stepsize, num_tests, eval_stepsize, cores, f"{name}_ucb16", start_str, walk_str, 16)
print("Upper confidence bound done.")
