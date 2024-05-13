# On the development of a practical Bayesian optimisation algorithm for expensive experiments and simulations with changing environmental conditions
Data and code associated with the paper "On the development of a practical
Bayesian optimisation algorithm for expensive experiments and simulations with
changing environmental conditions" currently in review.

**Abstract:** Experiments in engineering are typically conducted in controlled
environments where parameters can be set to any desired value. This assumes
that the same applies in a real-world setting, which is often incorrect as many
experiments are influenced by uncontrollable environmental conditions such as
temperature, humidity and wind speed. When optimising such experiments, the
focus should be finding optimal values conditionally on these uncontrollable
variables. This article extends Bayesian optimisation to the optimisation of
systems in changing environments that include controllable and uncontrollable
parameters. The extension fits a global surrogate model over all controllable
and environmental variables but optimises only the controllable parameters
conditional on measurements of the uncontrollable variables. The method is
validated on two synthetic test functions, and the effects of the noise level,
the number of environmental parameters, the parameter fluctuation, the
variability of the uncontrollable parameters, and the effective domain size are
investigated. ENVBO, the proposed algorithm from this investigation, is applied
to a wind farm simulator with eight controllable and one environmental
parameter. ENVBO finds solutions for the entire domain of the environmental
variable that outperform results from optimisation algorithms that only focus
on a fixed environmental value in all but one case while using a fraction of
their evaluation budget. This makes the proposed approach very sample-efficient
and cost-effective. An off-the-shelf open-source version of ENVBO is available
via the NUBO Python package.

**Cite:**

**Link:**
