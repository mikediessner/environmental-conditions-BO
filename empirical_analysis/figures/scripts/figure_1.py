# imports for Bayesian optimisation
import torch
import numpy as np
from nubo.acquisition import ExpectedImprovement
from nubo.models import GaussianProcess, fit_gp
from nubo.optimisation import lbfgsb
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import Interval

# imports for plotting
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter


# set seed for reproducibility
torch.manual_seed(1)


# axis formatter
class ScalarFormatterClass(ScalarFormatter):
   def _set_format(self):
      self.format = "%.1f"


# objective function
def func(x):
    y = x * torch.sin(x)
    return y.squeeze()


# input space
dims = 1
bounds = torch.tensor([[0.], [10.]])

# training data
x_train = torch.tensor([[1.], [5.], [9.5]], dtype=torch.double)
y_train = func(x_train)

# Bayesian optimisation loop
iters = 8

# plot properties
plt.rcParams.update({"font.size": 9, "font.family": "sans-serif", "font.sans-serif": "Arial"})

# make figure
fig, axs = plt.subplots(4, 2, figsize=(8.3*0.9, 11.7*0.9))

for iter in range(iters):
    
    # specify Gaussian process
    likelihood = GaussianLikelihood(noise_constraint=Interval(0.0, 1e-10))
    gp = GaussianProcess(x_train, y_train, likelihood=likelihood)
    
    # fit Gaussian process
    fit_gp(x_train, y_train, gp=gp, likelihood=likelihood, lr=0.1, steps=100)

    # specify acquisition function
    acq = ExpectedImprovement(gp=gp, y_best=torch.max(y_train))

    # optimise acquisition function
    x_new, _ = lbfgsb(func=acq, bounds=bounds, num_starts=5, num_samples=100)

    # evaluate new point
    y_new = func(x_new)
    
    # add to data
    x_train = torch.vstack((x_train, x_new))
    y_train = torch.hstack((y_train, y_new))
    
    # compute GP mean and variance for plotting
    x_plot = torch.linspace(start=-0.5, end=10.5, steps=1001, dtype=torch.double)
    gp.eval()
    pred = gp(x_plot)
    gp_means = pred.mean
    gp_vars = pred.variance.clamp_min(1e-10)
    gp_ci_upper = gp_means + torch.sqrt(gp_vars)*1.96
    gp_ci_lower = gp_means - torch.sqrt(gp_vars)*1.96

    # true objective function
    truth = func(x_plot)

    # acquisition function
    acq_func = torch.zeros(x_plot.size(0))
    for i in range(x_plot.size(0)):
        acq_func[i] = -acq(torch.reshape(x_plot[i], (1, 1)))

    # torch to numpy for plotting with matplotlib
    x_plot = x_plot.detach().numpy()
    gp_means = gp_means.detach().numpy()
    gp_ci_upper = gp_ci_upper.detach().numpy()
    gp_ci_lower = gp_ci_lower.detach().numpy()
    acq_func = acq_func.detach().numpy()
    
    row = int(iter/2)
    col = int(iter%2)

    # make plot
    axs[row, col].plot(x_train[:-1], y_train[:-1], marker="o", linewidth=0, color="navy", label="Observations", zorder=6)
    axs[row, col].axvline(float(x_train[-1]), color="red", linestyle="dotted", linewidth=1, label="Candidate", zorder=5)
    axs[row, col].plot(x_plot, gp_means, color="firebrick", label="Prediction", zorder=4)
    axs[row, col].plot(x_plot, truth, color="black", linestyle="dashed", label="Truth", zorder=2)
    axs[row, col].fill_between(x_plot, gp_ci_upper, gp_ci_lower, color="skyblue", label="Uncertainty", zorder=1)

    # second axis for expected improvement
    twin = axs[row, col].twinx()
    twin.fill_between(x_plot, acq_func, color="darkorange", label="EI", zorder=3, alpha=0.7)

    # set labels
    axs[row, col].set_title(f"Iteration {iter+1}")
    axs[row, col].set_ylabel("Output")
    twin.set_ylabel("Expected improvement")
    axs[row, col].set_xlabel("Input")
    
    # format axes
    axs[row, col].set_xlim((-0.5, 10.5))
    axs[row, col].set_ylim((-12., 10.))
    twin.set_ylim(0., np.max(acq_func)*3)
    formatter = ScalarFormatterClass(useMathText=True)
    formatter.set_powerlimits((0, 0))
    twin.yaxis.set_major_formatter(formatter)


# make custom legend
legend_elements = [Line2D([0], [0], marker="o", linewidth=0, color="navy", label="Observations"),
                   Patch(color="darkorange", label="Acquisition"),
                   Line2D([0], [0], color="firebrick", label="Prediction"),
                   Line2D([0], [0], color="red", linestyle="dotted", linewidth=1, label="Candidate"),
                   Patch(color="skyblue", label="Uncertainty"),
                   Line2D([0], [0], color="black", linestyle="dashed", label="Truth")]

# legend
fig.legend(handles=legend_elements, ncol=3, loc="lower center")
plt.tight_layout()
plt.subplots_adjust(bottom=0.09)


#################
## Save figure ##
#################

fig.set_rasterized(True)
plt.savefig("empirical_analysis/figures/Figure1.png")
plt.savefig("empirical_analysis/figures/Figure1.eps", format="eps")
plt.clf()
