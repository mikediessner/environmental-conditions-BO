import numpy as np
from scipy.stats import mannwhitneyu


def test(data1, data2):
    stat, p = mannwhitneyu(data1, data2)
    return f"Statistics={stat:.3f}, p={p:.3f}"


def final_mape(filepath, runs):
    mapes = []
    for run in range(runs):
        mape = np.loadtxt(f"{filepath}_mape_run{run+1}.csv", skiprows=1, delimiter=",")[-1, 1]
        mapes.append(mape)
    
    return np.array(mapes).reshape((-1, 1))


##########
## Levy ##
##########

bm = final_mape("empirical-analysis/results/levy_benchmark/levy_benchmark", 30).reshape((-1,))
ei = final_mape("empirical-analysis/results/levy_ei/levy_ei", 30).reshape((-1,))
logei = final_mape("empirical-analysis/results/levy_logei/levy_logei", 30).reshape((-1,))
ucb = final_mape("empirical-analysis/results/levy_ucb8/levy_ucb8", 30).reshape((-1,))

print("LEVY")
print("====")
print("EI: ", test(bm, ei))
print("LogEI: ", test(bm, logei))
print("UCB: ", test(bm, ucb))


##############
## Hartmann ##
##############

bm = final_mape("empirical-analysis/results/hartmann_benchmark/hartmann_benchmark", 30).reshape((-1,))
ei = final_mape("empirical-analysis/results/hartmann_ei/hartmann_ei", 30).reshape((-1,))
logei = final_mape("empirical-analysis/results/hartmann_logei/hartmann_logei", 30).reshape((-1,))
ucb = final_mape("empirical-analysis/results/hartmann_ucb8/hartmann_ucb8", 30).reshape((-1,))

print("HARTMANN")
print("========")
print("EI: ", test(bm, ei))
print("LogEI: ", test(bm, logei))
print("UCB: ", test(bm, ucb))
