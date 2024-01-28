import numpy as np
import matplotlib.pyplot as plt


def mean_absolute_error(y_true, y_pred):
    return np.sum(np.abs(y_true-y_pred))/y_true.shape[0]


def give_scores(iters, eval_stepsize, run, filename):

    truth = np.loadtxt(f"{filename}_truth_run{run+1}.csv", skiprows=1, delimiter=",")
    
    evaluations = list(range(0, iters+1, eval_stepsize))
    evaluations[0] = 2

    maes = np.zeros((len(evaluations), 2))
    for i, evaluation in enumerate(evaluations):
        pred = np.loadtxt(f"{filename}_pred_run{run+1}_eval{evaluation}.csv", skiprows=1, delimiter=",")
        mae = mean_absolute_error(truth[:, -1], pred[:, -1])
        maes[i, 0] = evaluation
        maes[i, 1] = mae
    
    np.savetxt(f"{filename}_mae_run{run+1}.csv",
        maes,
        delimiter=",",
        header="evaluation,mae",
        comments='')


def aggregate_mae(num_runs, filename):

    # load RSME for each run
    evals = np.loadtxt(f"{filename}_mae_run1.csv", skiprows=1, delimiter=",")[:, 0]
    maes = np.zeros((num_runs, evals.shape[0]))

    for run in range(num_runs):
        maes[run, :] = np.loadtxt(f"{filename}_mae_run{run+1}.csv", skiprows=1, delimiter=",")[:, -1]

    # compute mean and standard error over all runs
    means = np.mean(maes, axis=0)
    stds = np.std(maes, axis=0)
    
    out = np.hstack([evals.reshape(-1, 1),
                     means.reshape(-1, 1),
                     stds.reshape(-1, 1)])
    
    # save results
    np.savetxt(f"{filename}_maes.csv",
               out,
               header="evaluation,mean,std")

    return out


def make_plot(maes, filename):

    # plot properties
    plt.rcParams.update({"font.size": 9, "font.family": "sans-serif", "font.sans-serif": "Arial"})
    plt.tight_layout()

    # make figure
    fig = plt.figure()

    # plot
    plt.plot(maes[:, 0], maes[:, 1], label="Mean")
    plt.fill_between(maes[:, 0], maes[:, 1]-1.96*maes[:, 2], maes[:, 1]+1.96*maes[:, 2], alpha=0.25, label="95% confidence interval")
    
    # set axis limits
    plt.xlim(0, maes[-1, 0])
    plt.ylim(0, )

    # set axis labels
    plt.xlabel("Evaluations")
    plt.ylabel("Mean absolute error")

    # make legend
    plt.legend()

    # save plot
    fig.set_rasterized(True)
    plt.savefig(f"{filename}-mae.png", dpi=1000)
    # plt.savefig(f"{filename}-mae.eps", format="eps")
    plt.clf()


def final_mae(filepath, runs):
    maes = []
    for run in range(runs):
        mae = np.loadtxt(f"{filepath}_mae_run{run+1}.csv", skiprows=1, delimiter=",")[-1, 1]
        maes.append(mae)
    
    return np.array(maes).reshape((-1, 1))