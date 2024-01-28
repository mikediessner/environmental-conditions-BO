import torch
import numpy as np
import json
from nubo.models import GaussianProcess, fit_gp
from nubo.utils import gen_inputs
from gpytorch.likelihoods import GaussianLikelihood
from source.optim import optim
from source.helper import reduce_dims, make_header


def make_predictions(func, num_tests, eval_stepsize, filename, run):

    ##########################
    ## Algorithm parameters ##
    ##########################

    # load experiment parameters
    with open(f"{filename}_params_run{run+1}.txt") as json_file:
        params = json.load(json_file)
        bounds = torch.tensor(params["bounds"]).reshape((2, -1))
        dynamic_dims = params["dynamic_dims"]
        num_starts = params["num_starts"]
        num_samples = params["num_samples"]
    
    # gather all algorithm parameters
    params["num_tests"] = num_tests
    params["eval_stepsize"] = eval_stepsize

    # save prediction parameters
    with open(f"{filename}_predict_params_run{run+1}.txt", "w") as convert_file:
        convert_file.write(json.dumps(params, indent=4))


    ######################################################
    ## Find searched space of uncontrollable parameters ##
    ######################################################

    # get number of inputs
    dims = bounds.size((1))

    control_dims = list(set(range(dims)).difference(set(dynamic_dims)))
    control_bounds = bounds[:, control_dims]

    # load results
    results = torch.from_numpy(np.loadtxt(f"{filename}_run{run+1}.csv",
                                            skiprows=1,
                                            delimiter=","))

    # sample uncontrollable parameter values
    dynamic_results = results[:, dynamic_dims]
    x_min = torch.min(dynamic_results, dim=0).values
    x_max = torch.max(dynamic_results, dim=0).values
    x_min_max = torch.vstack([x_min.reshape(1, -1), x_max.reshape(1, -1)])
    x_test = gen_inputs(num_tests, len(dynamic_dims), x_min_max, 10)

    #####################
    ## Find true bests ##
    #####################

    # find truth for comparison
    true = np.zeros((num_tests, dims+1))
    for test in range(num_tests):

        x_new, y_new = optim(func=reduce_dims,
                             bounds=control_bounds,
                             num_samples=num_samples,
                             num_starts=num_starts,
                             func_args=(func, dynamic_dims, x_test[test, :], True))

        for i, dim in enumerate(dynamic_dims):
            x_new = torch.hstack([x_new[:, :dim], x_test[test, i].reshape(1, 1), x_new[:, dim:]])

        true[test, :-1], true[test, -1] = x_new, y_new
    
    # save truths
    np.savetxt(f"{filename}_truth_run{run+1}.csv",
        true,
        delimiter=",",
        header=make_header(dims, dynamic_dims),
        comments='')
    

    #########################################
    ## Predict bests with Gaussian process ##
    #########################################
        
    # predict with GP for comparison over number of evaluations
    if isinstance(eval_stepsize, int):
        evaluations = list(range(0, results.size(0)+1, eval_stepsize))
        evaluations[0] = 2
    elif isinstance(eval_stepsize, list):
        evaluations = eval_stepsize
    else:
        raise TypeError("eval_stepsize must be int or list of ints.")
       
    for evaluation in evaluations:
        
        # select training data
        x_train = results[:evaluation, :-1]
        y_train = results[:evaluation, -1]

        # fit Gaussian process
        likelihood = GaussianLikelihood()
        gp = GaussianProcess(x_train, y_train, likelihood=likelihood)
        fit_gp(x_train, y_train,
            gp=gp,
            likelihood=likelihood,
            lr=0.1,
            steps=200)
        gp.eval()
        
        # make predictions
        preds = np.zeros((num_tests, dims+1))
        for test in range(num_tests):
            
            def predict(x):
                pred = gp(x)
                mean = pred.mean.detach()
                return mean
            
            x_new, y_new = optim(func=reduce_dims,
                                 bounds=control_bounds,
                                 num_samples=num_samples,
                                 num_starts=num_starts,
                                 func_args=(predict, dynamic_dims, x_test[test, :], True))
            
            for i, dim in enumerate(dynamic_dims):
                x_new = torch.hstack([x_new[:, :dim], x_test[test, i].reshape(1, 1), x_new[:, dim:]])

            preds[test, :-1], preds[test, -1] = x_new, y_new

        # save predictions
        np.savetxt(f"{filename}_pred_run{run+1}_eval{evaluation}.csv",
                preds,
                delimiter=",",
                header=make_header(dims, dynamic_dims),
                comments='')
