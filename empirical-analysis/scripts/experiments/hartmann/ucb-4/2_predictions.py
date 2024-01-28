from source.predict import make_predictions
from nubo.test_functions import Hartmann6D
from joblib import Parallel, delayed


# set up experiment
func = Hartmann6D(minimise=False)     # objective function
num_tests = 25                        # number of test points
eval_stepsize = 10                    # increments of predictions
runs = 30                             # number of replicaions
directory = "results/hartmann/ucb-4/" # directory path
filename = "ucb-4-hartmann"           # filename for output csv files
cores = 6                             # number of parallel jobs

# run experiments in parallel
Parallel(n_jobs=cores)(delayed(make_predictions)(func=func,
                                                 num_tests=num_tests,
                                                 eval_stepsize=eval_stepsize,
                                                 filename=directory+filename,
                                                 run=run
                                                 ) for run in range(runs))
