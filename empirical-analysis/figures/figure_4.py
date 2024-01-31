import numpy as np
import matplotlib.pyplot as plt
from source.plot import give_scores, aggregate_mae, final_mae


# plot properties
plt.rcParams.update({"font.size": 9, "font.family": "sans-serif", "font.sans-serif": "Arial"})

# make figure
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8.3, 11.7*0.5))

# set parameters
evals = 100
eval_stepsize = 10
runs = 30
ylim = 3.5


##############################
## PLOT 1: Levy Performance ##
##############################

# compute mean absolute error
for run in range(runs):
    give_scores(evals, eval_stepsize, run, "data/levy/benchmark/benchmark-levy")
for run in range(runs):
    give_scores(evals, eval_stepsize, run, "data/levy/ei/ei-levy")
for run in range(runs):
    give_scores(evals, eval_stepsize, run, "data/levy/logei/logei-levy")
for run in range(runs):
    give_scores(evals, eval_stepsize, run, "data/levy/ucb-8/ucb-8-levy")

# compute average and standard deviation of mean absolute error over all runs
bm_maes = aggregate_mae(runs, "data/levy/benchmark/benchmark-levy")
ei_maes = aggregate_mae(runs, "data/levy/ei/ei-levy")
logei_maes = aggregate_mae(runs, "data/levy/logei/logei-levy")
ucb_maes = aggregate_mae(runs, "data/levy/ucb-8/ucb-8-levy")

# plot
axs[0, 0].plot(bm_maes[:, 0], bm_maes[:, 1], color="black", label="Random", zorder=7)
axs[0, 0].plot(ei_maes[:, 0], ei_maes[:, 1], color="tab:blue", label="EI", zorder=6)
axs[0, 0].fill_between(ei_maes[:, 0], ei_maes[:, 1]-1.96*ei_maes[:, 2], ei_maes[:, 1]+1.96*ei_maes[:, 2], color="tab:blue", alpha=0.25, zorder=3)
axs[0, 0].plot(logei_maes[:, 0], logei_maes[:, 1], color="tab:orange", label="LogEI", zorder=5)
axs[0, 0].fill_between(logei_maes[:, 0], logei_maes[:, 1]-1.96*logei_maes[:, 2], logei_maes[:, 1]+1.96*logei_maes[:, 2], color="tab:orange", alpha=0.25, zorder=2)
axs[0, 0].plot(ucb_maes[:, 0], ucb_maes[:, 1], color="tab:green", label="UCB", zorder=4)
axs[0, 0].fill_between(ucb_maes[:, 0], ucb_maes[:, 1]-1.96*ucb_maes[:, 2], ucb_maes[:, 1]+1.96*ucb_maes[:, 2], color="tab:green", alpha=0.25, zorder=1)
axs[0, 0].set_xlim(0, ei_maes[-1, 0])
axs[0, 0].set_ylim(0, )
axs[0, 0].set_xlabel("Evaluations")
axs[0, 0].set_ylabel("Mean absolute error")
axs[0, 0].legend(loc="upper right")


##################################
## PLOT 2: Hartmann Performance ##
##################################

# compute mean absolute error
for run in range(runs):
    give_scores(evals, eval_stepsize, run, "data/hartmann/benchmark/benchmark-hartmann")
for run in range(runs):
    give_scores(evals, eval_stepsize, run, "data/hartmann/ei/ei-hartmann")
for run in range(runs):
    give_scores(evals, eval_stepsize, run, "data/hartmann/logei/logei-hartmann")
for run in range(runs):
    give_scores(evals, eval_stepsize, run, "data/hartmann/ucb-8/ucb-8-hartmann")

# compute average and standard deviation of mean absolute error over all runs
bm_maes = aggregate_mae(runs, "data/hartmann/benchmark/benchmark-hartmann")
ei_maes = aggregate_mae(runs, "data/hartmann/ei/ei-hartmann")
logei_maes = aggregate_mae(runs, "data/hartmann/logei/logei-hartmann")
ucb_maes = aggregate_mae(runs, "data/hartmann/ucb-8/ucb-8-hartmann")

# plot
axs[0, 1].plot(bm_maes[:, 0], bm_maes[:, 1], color="black", label="Random", zorder=7)
axs[0, 1].plot(ei_maes[:, 0], ei_maes[:, 1], color="tab:blue", label="EI", zorder=6)
axs[0, 1].fill_between(ei_maes[:, 0], ei_maes[:, 1]-1.96*ei_maes[:, 2], ei_maes[:, 1]+1.96*ei_maes[:, 2], color="tab:blue", alpha=0.25, zorder=3)
axs[0, 1].plot(logei_maes[:, 0], logei_maes[:, 1], color="tab:orange", label="LogEI", zorder=5)
axs[0, 1].fill_between(logei_maes[:, 0], logei_maes[:, 1]-1.96*logei_maes[:, 2], logei_maes[:, 1]+1.96*logei_maes[:, 2], color="tab:orange", alpha=0.25, zorder=2)
axs[0, 1].plot(ucb_maes[:, 0], ucb_maes[:, 1], color="tab:green", label="UCB", zorder=4)
axs[0, 1].fill_between(ucb_maes[:, 0], ucb_maes[:, 1]-1.96*ucb_maes[:, 2], ucb_maes[:, 1]+1.96*ucb_maes[:, 2], color="tab:green", alpha=0.25, zorder=1)
axs[0, 1].set_xlim(0, ei_maes[-1, 0])
axs[0, 1].set_ylim(0, )
axs[0, 1].set_xlabel("Evaluations")
axs[0, 1].set_ylabel("Mean absolute error")
axs[0, 1].legend(loc="upper right")


#############################
## PLOT 3: Levy Difference ##
#############################

bm = final_mae("data/levy/benchmark/benchmark-levy", 30).reshape((-1,))
ei = final_mae("data/levy/ei/ei-levy", 30).reshape((-1,))
logei = final_mae("data/levy/logei/logei-levy", 30).reshape((-1,))
ucb = final_mae("data/levy/ucb-8/ucb-8-levy", 30).reshape((-1,))

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

bm = final_mae("data/hartmann/benchmark/benchmark-hartmann", 30).reshape((-1,))
ei = final_mae("data/hartmann/ei/ei-hartmann", 30).reshape((-1,))
logei = final_mae("data/hartmann/logei/logei-hartmann", 30).reshape((-1,))
ucb = final_mae("data/hartmann/ucb-8/ucb-8-hartmann", 30).reshape((-1,))

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
plt.savefig("Figure4.png")
plt.savefig("Figure4.eps", format="eps")
plt.clf()
