import numpy as np
import matplotlib.pyplot as plt


def mean_absolute_percentage_error(y_true, y_pred):
    return np.sum(np.abs((y_true-y_pred)/y_true))/y_true.shape[0]


def give_scores(iters, eval_stepsize, run, filename):

    truth = np.loadtxt(f"{filename}_truth_run{run+1}.csv", skiprows=1, delimiter=",")
    
    evaluations = list(range(0, iters+1, eval_stepsize))
    evaluations[0] = 2

    mapes = np.zeros((len(evaluations), 2))
    for i, evaluation in enumerate(evaluations):
        pred = np.loadtxt(f"{filename}_pred_run{run+1}_eval{evaluation}.csv", skiprows=1, delimiter=",")
        mape = mean_absolute_percentage_error(truth[:, -1], pred[:, -1])
        mapes[i, 0] = evaluation
        mapes[i, 1] = mape
    
    np.savetxt(f"{filename}_mape_run{run+1}.csv",
        mapes,
        delimiter=",",
        header="evaluation,mape",
        comments='')


def aggregate_mape(num_runs, filename):

    # load RSME for each run
    evals = np.loadtxt(f"{filename}_mape_run1.csv", skiprows=1, delimiter=",")[:, 0]
    mapes = np.zeros((num_runs, evals.shape[0]))

    for run in range(num_runs):
        mapes[run, :] = np.loadtxt(f"{filename}_mape_run{run+1}.csv", skiprows=1, delimiter=",")[:, -1]

    # compute mean and standard error over all runs
    means = np.mean(mapes, axis=0)
    stds = np.std(mapes, axis=0)
    
    out = np.hstack([evals.reshape(-1, 1),
                     means.reshape(-1, 1),
                     stds.reshape(-1, 1)])
    
    # save results
    np.savetxt(f"{filename}_mapes.csv",
               out,
               header="evaluation,mean,std")

    return out


def make_plot(mapes, filename):

    # plot properties
    plt.rcParams.update({"font.size": 9, "font.family": "sans-serif", "font.sans-serif": "Arial"})
    plt.tight_layout()

    # make figure
    fig = plt.figure()

    # plot
    plt.plot(mapes[:, 0], mapes[:, 1], label="Mean")
    plt.fill_between(mapes[:, 0], mapes[:, 1]-1.96*mapes[:, 2], mapes[:, 1]+1.96*mapes[:, 2], alpha=0.25, label="95% confidence interval")
    
    # set axis limits
    plt.xlim(0, mapes[-1, 0])
    plt.ylim(0, )

    # set axis labels
    plt.xlabel("Evaluations")
    plt.ylabel("Mean absolute percentage error")

    # make legend
    plt.legend()

    # save plot
    # fig.set_rasterized(True)
    plt.savefig(f"{filename}.png", dpi=1000)
    # plt.savefig(f"{filename}.eps", format="eps")
    plt.clf()

def final_mape(filepath, runs):
    mapes = []
    for run in range(runs):
        mape = np.loadtxt(f"{filepath}_mape_run{run+1}.csv", skiprows=1, delimiter=",")[-1, 1]
        mapes.append(mape)
    
    return np.array(mapes).reshape((-1, 1))

