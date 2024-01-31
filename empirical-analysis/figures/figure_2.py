import torch
import numpy as np
import matplotlib.pyplot as plt
from nubo.models import GaussianProcess, fit_gp
from nubo.utils import gen_inputs
from gpytorch.likelihoods import GaussianLikelihood
from source.test_functions import camel
from nubo.acquisition import ExpectedImprovement
from nubo.optimisation import single
from gpytorch.constraints import Interval


# plot properties
plt.rcParams.update({"font.size": 9, "font.family": "sans-serif", "font.sans-serif": "Arial"})

# make figure
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8.3, 11.7*0.5))


#####################################
## PLOT 1: Six-Hump Camel function ##
#####################################

# get data for plot
n = 401

# surface data
x = np.linspace(-2., 2., n)
y = np.linspace(-1., 1., n)

X, Y = np.meshgrid(x, y)
X, Y = np.ravel(X).reshape(-1, 1), np.ravel(Y).reshape(-1, 1)
XY = np.hstack([X, Y])
Z = camel(XY)

# plot
axs[0, 0].imshow(Z.reshape((n, n)), extent=[-2, 2, -1, 1], aspect="auto", origin="lower")
axs[0, 0].set_xlabel("Uncontrollable input")
axs[0, 0].set_ylabel("Controllabe input")



###########################################################
## PLOT 2: Gaussian process with highlighted measurement ##
###########################################################

# training data
bounds = torch.tensor([[-2., -1.], [2., 1.]])
torch.manual_seed(1)
x_train = gen_inputs(20, 2, bounds=bounds)
y_train = camel(x_train)

# Gaussian process
likelihood = GaussianLikelihood()
gp = GaussianProcess(x_train, y_train, likelihood)
fit_gp(x_train, y_train, gp, likelihood)

gp.eval()

# get data for plot
n = 101
conditional = -0.5

# surface data
x1 = np.linspace(-2., 2., n)
x2 = np.linspace(-1., 1., n)

x1, x2 = np.meshgrid(x1, x1)
x1, x2 = np.ravel(x1).reshape(-1, 1), np.ravel(x2).reshape(-1, 1)
x = np.hstack([x1, x2])
y = gp(torch.from_numpy(x)).mean.detach().numpy()

# plot
axs[0, 1].imshow(y.reshape((n, n)), extent=[-2, 2, -1, 1], aspect="auto", origin="lower")
axs[0, 1].axvline(x=conditional, linestyle="dashed", linewidth=1, color="red", label="Measurement", zorder=0)
axs[0, 1].scatter(x_train[:, 0].numpy(), x_train[:, 1].numpy(), marker="x", color="black", label="Training points", zorder=0)
axs[0, 1].set_xlabel("Uncontrollable input")
axs[0, 1].set_ylabel("Controllabe input")
axs[0, 1].legend()


########################################################
## PLOT 3: Gaussian process conditonal on measurement ##
########################################################

# data
x1_cond = np.ones(n) * conditional
x2 = np.linspace(-1., 1., n)
x_cond = np.hstack([x1_cond.reshape(-1, 1), x2.reshape(-1, 1)])
y_cond = gp(torch.from_numpy(x_cond)).mean.detach().numpy()

# plot
axs[1, 0].plot(x2, y_cond, color="firebrick", label="Prediction")
axs[1, 0].set_xlabel("Controllable input")
axs[1, 0].set_ylabel("Predicted output")
axs[1, 0].legend()


########################################
## PLOT 4: Bayesian optimisation step ##
########################################

# input space
bounds = torch.tensor([[-2., -1.], [2., 1.]]) # parameter bounds
dims = bounds.size(1)

# training data
torch.manual_seed(1)
x_train = gen_inputs(20, dims, bounds=bounds)
y_train = camel(x_train)

# specify Gaussian process
likelihood = GaussianLikelihood(noise_constraint=Interval(0.0, 1e-20))
gp = GaussianProcess(x_train, y_train, likelihood=likelihood)

# fit Gaussian process
fit_gp(x_train, y_train, gp=gp, likelihood=likelihood, lr=0.1, steps=200)

# specify acquisition function
acq = ExpectedImprovement(gp=gp, y_best=torch.max(y_train))

# optimise acquisition function
cons = {"type": "eq", "fun": lambda x: conditional - x[0]}

x_new, _ = single(func=acq,
                  method="SLSQP",
                  bounds=bounds,
                  constraints=cons,
                  num_starts=5)

# compute GP mean and variance for plotting
n = 101
x1_plot = torch.ones(n) * conditional
x2_plot = torch.linspace(start=-1.1, end=1.1, steps=n, dtype=torch.double)
x_plot = torch.hstack([x1_plot.reshape((-1, 1)), x2_plot.reshape((-1, 1))])
gp.eval()
pred = gp(x_plot)
gp_means = pred.mean
gp_vars = pred.variance.clamp_min(1e-10)
gp_ci_upper = gp_means + torch.sqrt(gp_vars)*1.96
gp_ci_lower = gp_means - torch.sqrt(gp_vars)*1.96

# acquisition function
acq_func = torch.zeros(x_plot.size(0))
for i in range(x_plot.size(0)):
    acq_func[i] = -acq(torch.reshape(x_plot[i], (1, -1)))

# torch to numpy for plotting with matplotlib
x_plot = x_plot.detach().numpy()
gp_means = gp_means.detach().numpy()
gp_ci_upper = gp_ci_upper.detach().numpy()
gp_ci_lower = gp_ci_lower.detach().numpy()
acq_func = acq_func.detach().numpy()

# make plot
axs[1, 1].plot(x2_plot, gp_means, color="firebrick", label="Prediction", zorder=2)
twin = axs[1, 1].twinx()
twin.fill_between(x2_plot, acq_func, color="darkorange", label="Acquisition", zorder=3, alpha=0.7)
axs[1, 1].fill_between(x2_plot, gp_ci_upper, gp_ci_lower, color="skyblue", label="Uncertainty", zorder=1)
axs[1, 1].axvline(float(x_new[:, 1]), color="red", linestyle="dotted", linewidth=1, label="Candidate", zorder=4)

# set labels
axs[1, 1].set_ylabel("Output")
axs[1, 1].set_xlabel("Controllable input")
twin.set_ylabel("Expected improvement")

# make legend
lines, labels = axs[1, 1].get_legend_handles_labels()
lines2, labels2 = twin.get_legend_handles_labels()
axs[1, 1].legend(lines + lines2, labels + labels2, ncol=2)

# format axes
axs[1, 1].set_xlim(-1.1, 1.1)
axs[1, 1].set_ylim(-3, 1.75)
twin.set_ylim(0., np.max(acq_func)*3)


#################
## Save figure ##
#################

plt.tight_layout()
fig.set_rasterized(True)
plt.savefig("Figure2.png")
plt.savefig("Figure2.eps", format="eps")
plt.clf()
