import torch
import numpy as np
from windfarm_simulator import simulate, simulate_aep, MIN_X, MAX_X, MIN_Y, MAX_Y 
from ENVBO import experiment, make_predictions
from helper import random_walk
from SLSQP import slsqp_benchmark
from BO import bo_benchmark


###################
## Problem Setup ##
###################

# Parameters
N_WT = 4                  # Number of wind turbines
N_EVALS = 200             # Number of simulator evaluations for ENVBO
MIN_WD, MAX_WD = 90, 135  # Lower and upper bounds of wind direction
MIN_WS, MAX_WS = 4, 10    # Lower and upper bounds of wind speed
WS = 6.0                  # Fixed wind speed
WDS = [90, 105, 120, 135] # Wind directions to benchmark

# Parameter bounds
bounds = torch.tensor([[MIN_X, ]*N_WT + [MIN_Y,]*N_WT + [MIN_WD, MIN_WS],
                       [MAX_X, ]*N_WT + [MAX_Y,]*N_WT + [MAX_WD, MAX_WS]])

# Environmental dimensions
env_dims = [2*N_WT, 2*N_WT+1]

# Result directory
DIR = "wind_farm_simulator/results"


#############################
## Optimisation with ENVBO ##
#############################

# Initial training data
x = np.random.uniform(MIN_X, MAX_X, N_WT)
y = np.random.uniform(MIN_Y, MAX_Y, N_WT)
wd = np.random.uniform(MIN_WD, MAX_WD, 1)
params = np.concatenate([x, y, wd, np.array([WS])])
obs = simulate_aep(params)
starts = torch.from_numpy(
    np.concatenate([params, np.array([obs])])
    ).reshape((1, -1))

# Random walk for wind direction
wd_values = random_walk(MIN_WD, MAX_WD, 5, N_EVALS-1, 16).reshape((-1, 1))
ws_values = np.ones((N_EVALS-1, 1))*WS
env_values = torch.from_numpy(np.hstack([wd_values, ws_values]))

np.savetxt(f"{DIR}/envbo_environmental_values.txt",
           env_values.numpy(), 
           delimiter=",",
           header="WD,WS",
           comments="")

# Run ENVBO
envbo_experiment = experiment(func=simulate_aep,
                              env_dims=env_dims,
                              env_values=env_values,
                              bounds=bounds,
                              evals=N_EVALS,
                              starts=starts,
                              num_starts=20,
                              num_samples=500)

np.savetxt(f"{DIR}/envbo_experiment_WT{N_WT}.txt",
           envbo_experiment.numpy(),
           header="X1,X2,X3,X4,Y1,Y2,Y3,Y4,WD,WS,AEP",
           delimiter=",",
           comments="")


##################
## Benchmarking ##
##################

envbo_results = np.zeros((len(WDS), bounds.shape[1]+1))
slsqp_results = np.zeros((len(WDS), bounds.shape[1]+2))
bo_results = np.zeros((len(WDS), bounds.shape[1]+2))

for i, wd in enumerate(WDS):
    env_values = torch.from_numpy(
        np.concatenate([np.array([wd]), np.array([WS])])
        )

    # Benchmark ENVBO  
    wt, _ = make_predictions(train_data=envbo_experiment,
                             env_dims=env_dims,
                             env_values=env_values,
                             bounds=bounds,
                             num_starts=20,
                             num_samples=500)
    wt_x = wt[0, :N_WT].reshape(N_WT)
    wt_y = wt[0, N_WT:2*N_WT].reshape(N_WT)
    res, site, wind_turbines = simulate(wt_x, wt_y, wd=wd, ws=WS)
    envbo_results[i, :] = np.concatenate(
        [wt.reshape(-1), np.array([res.aep().sum().values])]
        )

    np.savetxt(f"{DIR}/envbo_results_WT{N_WT}.txt",
               envbo_results,
               header="X1,X2,X3,X4,Y1,Y2,Y3,Y4,WD,WS,AEP",
               delimiter=",",
               comments="")

    # Benchmark SLSQP
    slsqp_x, slsqp_aep, slsqp_evals, slsqp_experiment = slsqp_benchmark(env_dims=env_dims,                   
                                                                        env_values=env_values, 
                                                                        bounds=bounds)
    slsqp_results[i, :] = np.concatenate(
        [slsqp_x.reshape(-1), -slsqp_aep.reshape(-1), slsqp_evals.reshape(-1)]
        )
    slsqp_wt_x = slsqp_x[0, :N_WT]
    slsqp_wt_y = slsqp_x[0, N_WT:2*N_WT]

    np.savetxt(f"{DIR}/slsqp_results_WT{N_WT}.txt",
               slsqp_results,
               header="X1,X2,X3,X4,Y1,Y2,Y3,Y4,WD,WS,AEP,Evals",
               delimiter=",",
               comments="")

    np.savetxt(f"{DIR}/slsqp_experiment_WT{N_WT}_WD{wd}.txt",
               slsqp_experiment,
               header="X1,X2,X3,X4,Y1,Y2,Y3,Y4,WD,WS,AEP",
               delimiter=",",
               comments="")

    # Benchmark BO
    starts[0, 8] = wd
    starts[0, 9] = WS
    bo_x, bo_aep, bo_evals, bo_experiment = bo_benchmark(env_dims=env_dims,
                                                         env_values=env_values,
                                                         bounds=bounds,
                                                         evals=50,
                                                         starts=starts,
                                                         num_starts=20,
                                                         num_samples=500)
    bo_wt_x = bo_x[0, :N_WT].reshape(N_WT)
    bo_wt_y = bo_x[0, N_WT:2*N_WT].reshape(N_WT)
    bo_res, _, _ = simulate(bo_wt_x, bo_wt_y, wd=wd, ws=WS)
    bo_results[i, :] = np.concatenate(
        [bo_x.reshape(-1),
         np.array([bo_res.aep().sum().values]),
         np.array([bo_evals])])

    np.savetxt(f"{DIR}/bo_results_WT{N_WT}.txt",
               bo_results,
               header="X1,X2,X3,X4,Y1,Y2,Y3,Y4,WD,WS,AEP,Evals",
               delimiter=",",
               comments="")

    np.savetxt(f"{DIR}/bo_experiment_WT{N_WT}_WD{wd}.txt",
               bo_experiment,
               header="X1,X2,X3,X4,Y1,Y2,Y3,Y4,WD,WS,AEP",
               delimiter=",",
               comments="")


############################
## Predictions with ENVBO ##
############################

# Compute ENVBO predictions for many different windspeeds
bounds = torch.tensor([[MIN_X, ]*N_WT + [MIN_Y,]*N_WT + [MIN_WD, MIN_WS],
                       [MAX_X, ]*N_WT + [MAX_Y,]*N_WT + [MAX_WD, MAX_WS]])

wds = np.linspace(90.0, 135.0, num=51, endpoint=True)
ws = np.array([WS])

envbo_results = np.zeros((len(wds), bounds.shape[1]+1))

for i, wd in enumerate(wds):
    env_values = torch.from_numpy(np.concatenate([np.array([wd]), ws]))

    # ENVBO
    wt, _ = make_predictions(train_data=envbo_experiment,
                             env_dims=env_dims,
                             env_values=env_values,
                             bounds=bounds,
                             num_starts=20,
                             num_samples=500)
    wt_x = wt[0, :N_WT].reshape(N_WT)
    wt_y = wt[0, N_WT:2*N_WT].reshape(N_WT)
    res, site, wind_turbines = simulate(wt_x, wt_y, wd=wd, ws=ws)
    envbo_results[i, :] = np.concatenate(
        [wt.reshape(-1), np.array([res.aep().sum().values])]
        )

np.savetxt(f"{DIR}/envbo_long_results_WT{N_WT}.txt",
            envbo_results,
            header="X1,X2,X3,X4,Y1,Y2,Y3,Y4,WD,WS,AEP",
            delimiter=",",
            comments="")
