Source code for the examples in "Robust Bayesian inference via coarsening".


Contents:
    autoregression - Autoregressive models of unknown order.
    bernoulli - Toy example: Perturbed Bernoulli trials.
    mixtures_simulation - Simulation example: Perturbed mixture of Gaussians.
    mixtures_flow_cytometry - Application: Robust clustering for flow cytometry.
    variable_selection_simulation - Simulation examples for variable selection in linear regression.
    variable_selection_cpp - Application: Effect of chemical exposures on birth weight.
    normal - Toy example: Perturbed Normal with outliers.


Most of the source code is written in the Julia language (https://julialang.org/), and a couple files are in R to compare with other methods (https://www.r-project.org/).  The code was implemented using Julia v0.6.0, so if you use a different version of Julia then tweaks might be needed.  The following Julia packages are required for full functionality: Distributions, PyPlot, JLD, and Lasso. These packages can be installed at the Julia command line by running:
    Pkg.add("Distributions")
    Pkg.add("Lasso")
    Pkg.add("JLD")
    ENV["PYTHON"]=""
    if (Pkg.installed("PyCall")!=nothing); Pkg.build("PyCall"); end
    Pkg.add("PyPlot")
    using PyPlot

To execute a given Julia file, say run.jl, cd to the appropriate folder, e.g., cd("../folder"), and do:
    include("run.jl")







