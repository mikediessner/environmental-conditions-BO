import torch
import numpy as np
import json
from nubo.models import GaussianProcess, fit_gp
from nubo.utils import gen_inputs
from nubo.algorithms import _cond_optim
from gpytorch.likelihoods import GaussianLikelihood
from source.helper import make_header


def make_predictions(func, num_tests, eval_stepsize, filename, run):

    ##########################
    ## Algorithm parameters ##
    ##########################

    # load experiment parameters
    with open(f"{filename}_params_run{run+1}.txt") as json_file:
        params = json.load(json_file)
        bounds = torch.tensor(params["bounds"]).reshape((2, -1))
        env_dims = params["env_dims"]
        num_starts = params["num_starts"]
        num_samples = params["num_samples"]
    
    # gather all algorithm parameters
    params["num_tests"] = num_tests
    params["eval_stepsize"] = eval_stepsize

    # save prediction parameters
    with open(f"{filename}_predict_params_run{run+1}.txt", "w") as convert_file:
        convert_file.write(json.dumps(params, indent=4))


    #############################################
    ## Test points of uncontrollable variables ##
    #############################################

    # get number of inputs
    dims = bounds.size((1))

    # load results
    results = torch.from_numpy(np.loadtxt(f"{filename}_run{run+1}.csv",
                                            skiprows=1,
                                            delimiter=","))

    # sample uncontrollable parameter values
    env_results = results[:, env_dims]
    x_min = torch.min(env_results, dim=0).values
    x_max = torch.max(env_results, dim=0).values
    x_min_max = torch.vstack([x_min.reshape(1, -1), x_max.reshape(1, -1)])
    x_test = gen_inputs(num_tests, len(env_dims), x_min_max, 10)

    #####################
    ## Find true bests ##
    #####################

    # find truth for comparison
    true = np.zeros((num_tests, dims+1))
    for test in range(num_tests):

        def neg_func(x):
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
                x = x.reshape((1, -1))
            return -func(x)

        x_new, y_new = _cond_optim(func=neg_func,
                                   env_dims=env_dims,
                                   env_values=x_test[test, :].tolist(),
                                   bounds=bounds,
                                   num_samples=num_samples,
                                   num_starts=num_starts)

        true[test, :-1], true[test, -1] = x_new, y_new
    
    # save truths
    np.savetxt(f"{filename}_truth_run{run+1}.csv",
        true,
        delimiter=",",
        header=make_header(dims, env_dims),
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
                if isinstance(x, np.ndarray):
                    x = torch.from_numpy(x)
                    x = x.reshape((1, -1))
                pred = gp(x)
                mean = pred.mean.detach()
                return -mean
            
            x_new, y_new = _cond_optim(func=predict,
                                       env_dims=env_dims,
                                       env_values=x_test[test, :].tolist(),
                                       bounds=bounds,
                                       num_samples=num_samples,
                                       num_starts=num_starts)
                        
            preds[test, :-1], preds[test, -1] = x_new, y_new

        # save predictions
        np.savetxt(f"{filename}_pred_run{run+1}_eval{evaluation}.csv",
                preds,
                delimiter=",",
                header=make_header(dims, env_dims),
                comments='')
