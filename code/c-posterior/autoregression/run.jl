# Apply c-posterior to autoregression model of variable order.
# Using analytical expression for evidence, assuming known variance.

# ____________________________________________________________________________________________________________
# Setup

# Settings
ns = [100,300,1000,3000,10000]  # sample sizes n to use
alphas = [Inf,250]  # alpha=Inf for standard posterior, alpha=250 for c-posterior chosen using calibration curve (below)
K = 60  # maximum value of k to use
Kshow = 20  # maximum value of k to show in plots
log_pK(k) = k*log(0.9) + log(0.1)  # log of prior on k, for k=0,1,2,...

# Data generation parameters
s0 = 1.0 # standard deviation of noise
sm = 0.5 # scale of misspecification
a0 = [0.25,0.25,-0.25,0.25] # autoregression coefficients
k0 = length(a0) # order of true auto regression model
f_noise(t) = randn()*s0 + sm*sin(t) # time-varying noise for the perturbation

# Model parameters
s = s0 # choose the model standard deviation to match the true value
sa = 1.0 # standard deviation of the prior on the autoregression coefficients

# ____________________________________________________________________________________________________________
# Helper functions

using PyPlot
using Distributions
draw() = (pause(0.001); get_current_fig_manager()[:window][:raise_]())
logsumexp(x) = (m=maximum(x); (m==-Inf ? -Inf : log(sum(exp.(x-m)))+m))
latex(s) = latexstring(replace(s," ","\\,\\,"))

# Code for autoregression model
include("AR.jl")

# Function to compute the log of the posterior probability of each k = 0,1,...,K
log_posterior(x,zeta) = (log_m = Float64[log_marginal(x,k,s,sa,zeta)+log_pK(k) for k=0:K]; log_m - logsumexp(log_m))

# ____________________________________________________________________________________________________________
# Calibration of alpha

if true
    srand(0)  # Reset RNG
    n = maximum(ns)  # sample size to use for calibration of alpha
    x = generate(n,a0,f_noise)  # generate data
    alpha_values = [10.^collect(1:0.1:5); Inf]  # values of alpha to consider
    zetas = (1/n) ./ (1/n + 1.0./alpha_values)  # corresponding powers
    log_m = Float64[log_marginal(x,k,s,sa,1.0) for k=0:K]  # standard log marginal likelihood (zeta=1.0)
    El = Float64[sum(log_m .* exp.(log_posterior(x,zeta))) for zeta in zetas]  # posterior expectation of log marginal likelihood, for each zeta
    Ek = Float64[sum((0:K).*exp.(log_posterior(x,zeta))) for zeta in zetas]  # posterior expectation of model order k, for each zeta

    # Plot the calibration curve
    figure(1,figsize=(8,2.5)); clf()
    subplots_adjust(bottom=0.2)
    X,Y,a = Ek[15],El[15],alpha_values[15]
    plot(X,Y,"rx",ms=10,mew=2)
    plot(Ek,El,"ko-",ms=2,lw=1)
    yl = ylim()
    text(X+1,Y+(yl[1]-yl[2])*0.10,latex("\\alpha=$(round(Int,a))"),fontsize=14)
    gca()[:ticklabel_format](style="sci",axis="y",scilimits=(0,0),fontsize=8)
    title(latex("\\mathrm{Calibration of} \\alpha"),fontsize=17)
    ylabel(latex("\\mathrm{E}_\\alpha\\mathrm{(loglik | data)}"),fontsize=15)
    xlabel(latex("\\mathrm{E}_\\alpha(k | \\mathrm{data})"),fontsize=16)
    savefig("AR-calibration-n=$n.png",dpi=150)
end

# ____________________________________________________________________________________________________________
# Plot some data

figure(2,figsize=(8,2.5)); clf()
subplots_adjust(bottom=0.2)
x = generate(200,a0,f_noise)
plot(x)
title(latex("\\mathrm{AR($k0) data with time-varying noise}"),fontsize=17)
xlabel(latex("t  \\mathrm{(time)}"),fontsize=16)
ylabel(L"x_t",fontsize=18)
ylim(-4,4)
draw()
savefig("AR-data.png",dpi=150)

# ____________________________________________________________________________________________________________
# Compute results and plot

# Initialize plots
figure(3,figsize=(8,2.5)); clf()
subplots_adjust(bottom=0.2)
title(latex("\\mathrm{Standard posterior}"),fontsize=17)
ylabel(latex("\\pi(k|\\mathrm{data})"),fontsize=15)
xlabel(latex("k  \\mathrm{(\\# of coefficients)}"),fontsize=16)
figure(4,figsize=(8,2.5)); clf()
subplots_adjust(bottom=0.2)
title(latex("\\mathrm{Coarsened posterior}"),fontsize=17)
ylabel(latex("\\pi(k|\\mathrm{data})"),fontsize=15)
xlabel(latex("k  \\mathrm{(\\# of coefficients)}"),fontsize=16)

srand(0) # Reset RNG
x_all = generate(maximum(ns),a0,f_noise) # Generate data
for (i_n,n) in enumerate(ns) # for each value of n
    x = x_all[1:n] # extract the first n data points in the sequence

    for (i_a,alpha) in enumerate(alphas) # for each value of alpha

        # Compute marginal likelihood for each k
        zeta = (1/n)/(1/n + 1/alpha) # power to raise the likelihood to
        log_m = Float64[log_marginal(x,k,s,sa,zeta) for k = 0:K]  # log_m[k+1] = log p(x|k) = (robustified) log marginal likelihood for model order k

        # Plot log marginal likelihood
        figure(i_a*10+i_n,figsize=(8,2.5)); clf()
        subplots_adjust(top=.8,bottom=0.25,left=.13)
        ks = 0:Kshow
        plot(ks,log_m[ks+1],"ko-",markersize=8,linewidth=2)
        xlim(0,Kshow)
        xticks(0:Kshow)
        mx,mn = maximum(log_m[ks+1]),minimum(log_m[ks+1])
        pad = (mx-mn)/20
        ylim(mn-pad,mx+pad)
        ticklabel_format(axis="y",style="sci",scilimits=(-4,4))
        if alpha<Inf; title(latex("\\mathrm{Coarsened}, n=$n"),fontsize=17)
        else; title(latex("\\mathrm{Standard}, n=$n"),fontsize=17)
        end
        xlabel(latex("k  \\mathrm{(\\# of coefficients)}"),fontsize=16)
        if n==ns[1]
            ylabel(latex("\\log p(\\mathrm{data}|k)"),fontsize=15)
        end
        draw()
        savefig("AR-alpha=$(alpha<Inf? repr(round(Int,alpha)) : alpha)-n=$n.png",dpi=150)

        # Plot posterior distribution on k
        figure(2+i_a)
        log_post = log_m + log_pK(0:K)
        posterior = exp.(log_post - logsumexp(log_post))
        colors = "bgyrm"
        shapes = "ds^v*"
        ks = 0:(alpha<Inf? Kshow:K)
        plot(ks,posterior[ks+1],"$(colors[i_n])$(shapes[i_n])-",label=latex("n=$n"),mec="k",ms=8,lw=2)
    end
end

# Clean up plots and save to file
figure(3)
xlim(0,40)
xticks(0:5:40)
ylim(0,1)
legend(numpoints=1,loc=9,fontsize=12)
draw()
savefig("AR-standard-posterior.png",dpi=150)

figure(4)
xlim(0,12)
xticks(0:12)
ylim(0,1)
legend(numpoints=1,loc="upper right",fontsize=12)
draw()
savefig("AR-coarsened-posterior.png",dpi=150)


nothing
