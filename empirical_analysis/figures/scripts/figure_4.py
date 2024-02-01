import numpy as np
import matplotlib.pyplot as plt
from helper import give_scores, aggregate_mape, final_mape


# plot properties
plt.rcParams.update({"font.size": 9, "font.family": "sans-serif", "font.sans-serif": "Arial"})

# make figure
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8.3, 11.7*0.5))

# set parameters
evals = 100
eval_stepsize = 10
runs = 30
ylim = 3.5

# results directory
DIR = "empirical_analysis/results"


##############################
## PLOT 1: Levy Performance ##
##############################

# compute mean absolute error
for run in range(runs):
    give_scores(evals, eval_stepsize, run, f"{DIR}/levy_benchmark/levy_benchmark")
for run in range(runs):
    give_scores(evals, eval_stepsize, run, f"{DIR}/levy_ei/levy_ei")
for run in range(runs):
    give_scores(evals, eval_stepsize, run, f"{DIR}/levy_logei/levy_logei")
for run in range(runs):
    give_scores(evals, eval_stepsize, run, f"{DIR}/levy_ucb8/levy_ucb8")

# compute average and standard deviation of mean absolute error over all runs
bm_mapes = aggregate_mape(runs, f"{DIR}/levy_benchmark/levy_benchmark")
ei_mapes = aggregate_mape(runs, f"{DIR}/levy_ei/levy_ei")
logei_mapes = aggregate_mape(runs, f"{DIR}/levy_logei/levy_logei")
ucb_mapes = aggregate_mape(runs, f"{DIR}/levy_ucb8/levy_ucb8")

# plot
axs[0, 0].plot(bm_mapes[:, 0], bm_mapes[:, 1], color="black", label="Random", zorder=7)
axs[0, 0].plot(ei_mapes[:, 0], ei_mapes[:, 1], color="tab:blue", label="EI", zorder=6)
axs[0, 0].fill_between(ei_mapes[:, 0], ei_mapes[:, 1]-1.96*ei_mapes[:, 2], ei_mapes[:, 1]+1.96*ei_mapes[:, 2], color="tab:blue", alpha=0.25, zorder=3)
axs[0, 0].plot(logei_mapes[:, 0], logei_mapes[:, 1], color="tab:orange", label="LogEI", zorder=5)
axs[0, 0].fill_between(logei_mapes[:, 0], logei_mapes[:, 1]-1.96*logei_mapes[:, 2], logei_mapes[:, 1]+1.96*logei_mapes[:, 2], color="tab:orange", alpha=0.25, zorder=2)
axs[0, 0].plot(ucb_mapes[:, 0], ucb_mapes[:, 1], color="tab:green", label="UCB", zorder=4)
axs[0, 0].fill_between(ucb_mapes[:, 0], ucb_mapes[:, 1]-1.96*ucb_mapes[:, 2], ucb_mapes[:, 1]+1.96*ucb_mapes[:, 2], color="tab:green", alpha=0.25, zorder=1)
axs[0, 0].set_xlim(0, ei_mapes[-1, 0])
axs[0, 0].set_ylim(0, )
axs[0, 0].set_xlabel("Evaluations")
axs[0, 0].set_ylabel("Mean absolute percentage error")
axs[0, 0].legend(loc="upper right")


##################################
## PLOT 2: Hartmann Performance ##
##################################

# compute mean absolute error
for run in range(runs):
    give_scores(evals, eval_stepsize, run, f"{DIR}/hartmann_benchmark/hartmann_benchmark")
for run in range(runs):
    give_scores(evals, eval_stepsize, run, f"{DIR}/hartmann_ei/hartmann_ei")
for run in range(runs):
    give_scores(evals, eval_stepsize, run, f"{DIR}/hartmann_logei/hartmann_logei")
for run in range(runs):
    give_scores(evals, eval_stepsize, run, f"{DIR}/hartmann_ucb8/hartmann_ucb8")

# compute average and standard deviation of mean absolute error over all runs
bm_mapes = aggregate_mape(runs, f"{DIR}/hartmann_benchmark/hartmann_benchmark")
ei_mapes = aggregate_mape(runs, f"{DIR}/hartmann_ei/hartmann_ei")
logei_mapes = aggregate_mape(runs, f"{DIR}/hartmann_logei/hartmann_logei")
ucb_mapes = aggregate_mape(runs, f"{DIR}/hartmann_ucb8/hartmann_ucb8")

# plot
axs[0, 1].plot(bm_mapes[:, 0], bm_mapes[:, 1], color="black", label="Random", zorder=7)
axs[0, 1].plot(ei_mapes[:, 0], ei_mapes[:, 1], color="tab:blue", label="EI", zorder=6)
axs[0, 1].fill_between(ei_mapes[:, 0], ei_mapes[:, 1]-1.96*ei_mapes[:, 2], ei_mapes[:, 1]+1.96*ei_mapes[:, 2], color="tab:blue", alpha=0.25, zorder=3)
axs[0, 1].plot(logei_mapes[:, 0], logei_mapes[:, 1], color="tab:orange", label="LogEI", zorder=5)
axs[0, 1].fill_between(logei_mapes[:, 0], logei_mapes[:, 1]-1.96*logei_mapes[:, 2], logei_mapes[:, 1]+1.96*logei_mapes[:, 2], color="tab:orange", alpha=0.25, zorder=2)
axs[0, 1].plot(ucb_mapes[:, 0], ucb_mapes[:, 1], color="tab:green", label="UCB", zorder=4)
axs[0, 1].fill_between(ucb_mapes[:, 0], ucb_mapes[:, 1]-1.96*ucb_mapes[:, 2], ucb_mapes[:, 1]+1.96*ucb_mapes[:, 2], color="tab:green", alpha=0.25, zorder=1)
axs[0, 1].set_xlim(0, ei_mapes[-1, 0])
axs[0, 1].set_ylim(0, )
axs[0, 1].set_xlabel("Evaluations")
axs[0, 1].set_ylabel("Mean absolute percentage error")
axs[0, 1].legend(loc="upper right")


#############################
## PLOT 3: Levy Difference ##
#############################

bm = final_mape(f"{DIR}/levy_benchmark/levy_benchmark", 30).reshape((-1,))
ei = final_mape(f"{DIR}/levy_ei/levy_ei", 30).reshape((-1,))
logei = final_mape(f"{DIR}/levy_logei/levy_logei", 30).reshape((-1,))
ucb = final_mape(f"{DIR}/levy_ucb8/levy_ucb8", 30).reshape((-1,))

# plot
axs[1, 0].hlines(y=0, xmin=0, xmax=4, linestyle="dashed", color="black")
# axs[1, 0].boxplot([bm-ei, bm-logei, bm-ucb])
axs[1, 0].violinplot(bm-ei, positions=[1], showmeans=True)
axs[1, 0].violinplot(bm-logei, positions=[2], showmeans=True)
axs[1, 0].violinplot(bm-ucb, positions=[3], showmeans=True)
axs[1, 0].set_ylabel("Difference")
axs[1, 0].set_xticks([1, 2, 3], labels=["EI", "LogEI", "UCB"])
axs[1, 0].set_xlim(0, 4)


#################################
## PLOT 4: Hartmann Difference ##
#################################

bm = final_mape(f"{DIR}/hartmann_benchmark/hartmann_benchmark", 30).reshape((-1,))
ei = final_mape(f"{DIR}/hartmann_ei/hartmann_ei", 30).reshape((-1,))
logei = final_mape(f"{DIR}/hartmann_logei/hartmann_logei", 30).reshape((-1,))
ucb = final_mape(f"{DIR}/hartmann_ucb8/hartmann_ucb8", 30).reshape((-1,))

# plot
axs[1, 1].hlines(y=0, xmin=0, xmax=4, linestyle="dashed", color="black")
# axs[1, 1].boxplot([bm-ei, bm-logei, bm-ucb])
axs[1, 1].violinplot(bm-ei, positions=[1], showmeans=True)
axs[1, 1].violinplot(bm-logei, positions=[2], showmeans=True)
axs[1, 1].violinplot(bm-ucb, positions=[3], showmeans=True)
axs[1, 1].set_ylabel("Difference")
axs[1, 1].set_xticks([1, 2, 3], labels=["EI", "LogEI", "UCB"])
axs[1, 1].set_xlim(0, 4)


#################
## Save figure ##
#################

plt.tight_layout()
fig.set_rasterized(True)
plt.savefig("empirical_analysis/figures/Figure4.png")
plt.savefig("empirical_analysis/figures/Figure4.eps", format="eps")
plt.clf()
