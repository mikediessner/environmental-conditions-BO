import pandas as pd
import numpy as np


def final_mae(filepath, runs):
    maes = []
    for run in range(runs):
        mae = np.loadtxt(f"{filepath}_mae_run{run+1}.csv", skiprows=1, delimiter=",")[-1, 1]
        maes.append(mae)
    
    return np.array(maes).reshape((-1, 1))

ei_obs = final_mae("results/hartmann/observation-ei/observation-ei-hartmann", 30)
ei_pred = final_mae("results/hartmann/prediction-ei/prediction-ei-hartmann", 30)
ei_samp = final_mae("results/hartmann/sampling-ei/sampling-ei-hartmann", 30)
ei_mc = final_mae("results/hartmann/montecarlo-ei/montecarlo-ei-hartmann", 30)

logei_obs = final_mae("results/hartmann/observation-logei/observation-logei-hartmann", 30)
logei_pred = final_mae("results/hartmann/prediction-logei/prediction-logei-hartmann", 30)
logei_samp = final_mae("results/hartmann/sampling-logei/sampling-logei-hartmann", 30)

ucb_4 = final_mae("results/hartmann/ucb-4/ucb-4-hartmann", 30)
ucb_8 = final_mae("results/hartmann/ucb-8/ucb-8-hartmann", 30)
ucb_16 = final_mae("results/hartmann/ucb-16/ucb-16-hartmann", 30)

comparison = pd.DataFrame(np.hstack([ei_obs, ei_pred, ei_samp, ei_mc, logei_obs, logei_pred, logei_samp, ucb_4, ucb_8, ucb_16]),
                          columns=["ei_obs", "ei_pred", "ei_samp", "ei_mc", "logei_obs", "logei_pred", "logei_samp", "ucb_4", "ucb_8", "ucb_16"])

comparison.to_csv("results/hartmann/comparison-mae.csv", index=False)