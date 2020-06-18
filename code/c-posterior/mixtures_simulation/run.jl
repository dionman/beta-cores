# Simulation to compare coarsened mixture to a standard mixture, using univariate normal components.
# ________________________________________________________________________________________________________
# Setup

# Settings
k0 = 2  # true number of components (must be 2 or 4)
if k0==2; alpha = 800; elseif k0==4; alpha = 2000; else; error("Unknown value of k0"); end  # choice of alpha based on calibration curve
ns = [200,1000,5000,10000,20000]  # sample sizes n to use
nreps = 5  # number of times to run the simulation
n_total = 10^4  # total number of MCMC iterations
n_burn = round(Int,n_total/10)  # number of MCMC iterations to discard as burn-in
n_init = round(Int,n_burn/2)  # number of MCMC iterations for initialization with periodic random splits
cutoff = 0.02  # nonnegligible cluster size: 100*cutoff %
from_file = false  # load previous results from file

# Model parameters
K = 20  # number of components
gamma0 = 1/(2*K)  # Dirichlet concentration parameter
mu0 = 0.0  # prior mean of component means
sigma0 = 5.0  # prior standard deviation of component means
a0,b0 = 1.0,1.0  # prior InverseGamma parameters for component variances

# ____________________________________________________________________________________________________________
# Helper functions

using Distributions
using PyPlot
using HDF5, JLD
drawnow() = (pause(0.001); get_current_fig_manager()[:window][:raise_]())
latex(s) = latexstring(replace(s," ","\\,\\,"))
logsumexp(x) = (m = maximum(x); m == -Inf ? -Inf : log.(sum(exp.(x-m))) + m)

# Code for mixture model MCMC algorithm
include("core.jl")

# ________________________________________________________________________________________________________
# Generating simulated data 

# True parameters
if k0==2; mu_t = [-2,2]; sigma_t = [0.7,0.8]; w_t = [0.5,0.5]; xmin,xmax = -4.5,4.5; ymax = 0.35; srand(4); end
if k0==4; mu_t = [-3.5,3,0,6]; sigma_t = [0.8,0.5,0.4,0.5]; w_t = [0.25,.25,.3,.2]; xmin,xmax = -6,8; ymax = 0.35; srand(3); end

# Randomly generate a perturbation using a Dirichlet process mixture model
L = 10000  # number of components for DP approximation
alpha0 = 500  # DP concentration parameter
s = 0.25  # component standard deviation
y_t = rand(Categorical(w_t),L)  # assignment of DP atoms to true components
beta_t = randn(L).*sigma_t[y_t] + mu_t[y_t]  # DP atom locations
p_t = rand(Dirichlet(L,alpha0/L))  # DP atom weights

# Function to generate simulated data from the observed data distribution (perturbation from true)
function generate_data(n)
    z_t = rand(Categorical(p_t),n)
    x = randn(n).*s + beta_t[z_t]
    return x
end

# Plot simulated data
if true
    n = 2000
    x = generate_data(n)
    figure(1,figsize=(8,2.5)); clf()
    subplots_adjust(bottom=0.2)
    xs = linspace(xmin,xmax,1000)
    density = Float64[sum(w_t.*normpdf(xi,mu_t,sigma_t)) for xi in xs]
    plot(xs,density,"b--",lw=2)
    density = Float64[sum(p_t.*normpdf(xi,beta_t,s)) for xi in xs]
    plot(xs,density,"r-",lw=1.5)
    counts,edges = histogram(x,linspace(xmin,xmax,50))
    dx = edges[2]-edges[1]
    #bar(edges[1:end-1],(counts/n)/dx,dx,color="w",edgecolor="k")
    title(latex("\\mathrm{Data distribution} (k_0=$k0)"),fontsize=17)
    xlabel(L"x",fontsize=16)
    ylabel(L"\mathrm{density}",fontsize=16)
    xlim(xmin,xmax)
    ylim(0,ymax)
    drawnow()
    savefig("mixsim-data-k=$k0.png",dpi=150)
end

# ________________________________________________________________________________________________________
# Choose alpha using calibration curve

if true
    srand(0)  # reset RNG
    n = 10000  # sample size to use for calibration curve
    x = generate_data(n)
    
    # Compute calibration curve
    if from_file
        # Load results from file
        Ek,El,alphas = load("mixsim-calibration-k=$k0-n=$n.jld","Ek","El","alphas")
    else
        alphas = [10.^collect(1:0.1:5); Inf]  # values of alpha to consider
        n_alphas = length(alphas)
        El = zeros(n_alphas)  # posterior expectation of the log likelihood, for each alpha
        Ek = zeros(n_alphas)  # posterior expectation of the number of non-negligible clusters, for each alpha
        for (i_a,a) in enumerate(alphas)
            println("alpha = $a")
            zeta = (1/n)/(1/n + 1/a)
            tic()
            N_r,p_r,mu_r,sigma_r,logw,L_r = sampler(x,n_total,n_init,K,gamma0,mu0,sigma0,a0,b0,zeta; mode="downweight")
            toc()
            use = n_burn+1:n_total
            El[i_a] = mean(L_r[use])
            kb_r = vec(sum(N_r.>n*cutoff, 1))
            Ek[i_a] = mean(kb_r[use])
        end
        # Save results to file
        save("mixsim-calibration-k=$k0-n=$n.jld","Ek",Ek,"El",El,"alphas",alphas)
    end

    # Plot calibration curve
    figure(2,figsize=(8,2.5)); clf()
    subplots_adjust(bottom=0.2)
    if k0==2
        X,Y,a = Ek[20],El[20],alphas[20]
        plot(X,Y,"rx",ms=10,mew=2)
        plot(Ek,El,"ko-",ms=2,lw=1)
        yl = ylim()
        ylim(yl[1],yl[2]+500)
        text(X-0.07,Y+(yl[1]-yl[2])*0.20,latex("\\alpha=$(round(Int,a))"),fontsize=14)
    elseif k0==4
        X,Y,a = Ek[24],El[24],alphas[24]
        plot(X,Y,"rx",ms=10,mew=2)
        plot(Ek,El,"ko-",ms=2,lw=1)
        yl = ylim()
        ylim(yl[1],yl[2]+500)
        text(X-0.35,Y+(yl[1]-yl[2])*0.23,latex("\\alpha=$(round(Int,a))"),fontsize=14)
    end
    gca()[:ticklabel_format](style="sci",axis="y",scilimits=(0,0),fontsize=8)
    yticks(fontsize=9)
    title(latex("\\mathrm{Calibration of} \\alpha  (k_0=$k0)"),fontsize=17)
    ylabel(latex("\\hat\\mathrm{E}_\\alpha\\mathrm{(loglik | data)}"),fontsize=15)
    xlabel(latex("\\hat\\mathrm{E}_\\alpha(k_{2\\%} | \\mathrm{data})"),fontsize=16,labelpad=0)
    savefig("mixsim-calibration-k=$k0-n=$n.png",dpi=150)
    drawnow()
end

# ________________________________________________________________________________________________________
# Run inference algorithms

# Function to plot estimated density and the individual mixture components
function plot_densities(fignum,titlestring,p,mu,sigma)
    #close(fignum)
    figure(fignum,figsize=(8,2.5)); clf()
    subplots_adjust(bottom=0.2)
    xs = linspace(xmin,xmax,1000)
    density = Float64[sum(p.*normpdf(xi,mu,sigma)) for xi in xs]
    plot(xs,density,"k--",linewidth=3)
    for i = 1:length(p)
        if p[i] > 0.001
            component = Float64[p[i].*normpdf(xi,mu[i],sigma[i]) for xi in xs]
            plot(xs,component,linewidth=1)
        end
    end
    xlim(xmin,xmax)
    ylim(0,0.4)
    title(latex("\\mathrm{$titlestring}"),fontsize=17)
    xlabel(L"x",fontsize=16)
    ylabel(latex("\\mathrm{density}"),fontsize=15)
end


# Run algorithms
srand(0)
use = (n_burn+1:n_total)
n_use = length(use)
nns = length(ns)
offsets = (0:4)*1000
if from_file
    k_posteriors = load("k_posteriors-k=$k0.jld","k_posteriors")
else
    k_posteriors = zeros(K+1,nreps,nns,3)
    for (i_n,n) in enumerate(ns)
        for rep in 1:nreps
            println("n=$n  rep=$rep")
            x = generate_data(n)

            # -------------------------- Standard mixture posterior -------------------------- 
            println("Running standard mixture...")
            tic()
            N_r,p_r,mu_r,sigma_r,logw,L_r  = sampler(x,n_total,n_init,K,gamma0,mu0,sigma0,a0,b0,1.0)
            toc()
            kb_r = vec(sum(N_r.>n*cutoff, 1))
            counts,~ = histogram(kb_r[use],-0.5:1:K+0.5)
            k_posteriors[:,rep,i_n,1] = counts/n_use
            if n==ns[end]
                for o in offsets
                    plot_densities(10,"Standard posterior",p_r[:,end-o],mu_r[:,end-o],sigma_r[:,end-o])
                    ylim(0,ymax)
                    savefig("mixsim-density-standard-k=$k0-n=$n-o=$o.png",dpi=150)
                end
            end

            # -------------------------- Coarsened mixture posterior -------------------------- 
            println("Running coarsened mixture...")
            zeta = alpha/(alpha + n)
            tic()
            N_r,p_r,mu_r,sigma_r,logw,L_r = sampler(x,n_total,n_init,K,gamma0,mu0,sigma0,a0,b0,zeta; mode="subset")
            toc()

            # compute weights for importance sampling
            w = exp.(logw[use] - logsumexp(logw[use]))
            cv = std(n_use*w)  # estimate coef of variation
            ess = 1/(1+cv^2)  # effective sample size
            @printf "ESS = %.3f%%\n" 100*ess

            kb_r = vec(sum(N_r.>n*zeta*cutoff, 1))
            pk,~ = histogram(kb_r[use],-0.5:1:K+0.5; weights=w)
            k_posteriors[:,rep,i_n,2] = pk
            if n==ns[end]
                for o in offsets
                    plot_densities(10,"Coarsened posterior",p_r[:,end-o],mu_r[:,end-o],sigma_r[:,end-o])
                    ylim(0,ymax)
                    savefig("mixsim-density-coarsened-k=$k0-n=$n-o=$o.png",dpi=150)
                end
            end

            # -------------------------- Conditional coarsening -------------------------- 
            println("Running conditional coarsening algorithm...")
            zeta = alpha/(alpha + n)
            tic()
            N_r,p_r,mu_r,sigma_r,logw,L_r  = sampler(x,n_total,n_init,K,gamma0,mu0,sigma0,a0,b0,zeta; mode="downweight")
            toc()
            kb_r = vec(sum(N_r.>n*cutoff, 1))
            counts,~ = histogram(kb_r[use],-0.5:1:K+0.5)
            k_posteriors[:,rep,i_n,3] = counts/n_use
            if n==ns[end]
                for o in offsets
                    plot_densities(10,"Conditional coarsening",p_r[:,end-o],mu_r[:,end-o],sigma_r[:,end-o])
                    ylim(0,ymax)
                    savefig("mixsim-density-conditional-k=$k0-n=$n-o=$o.png",dpi=150)
                end
            end
        end
    end
    save("k_posteriors-k=$k0.jld","k_posteriors",k_posteriors)
end


# ________________________________________________________________________________________________________
# Plot results

colors = "bgyrm"
shapes = "ds^v*"
Kshow = 12
# Plot posterior
function plot_k_posteriors(fignum,index,titlestring)
    figure(fignum,figsize=(8,2.5)); clf()
    subplots_adjust(bottom=0.2)
    for (i_n,n) in enumerate(ns)
        kp = vec(mean(k_posteriors[1:Kshow+1,:,i_n,index],2))
        plot(0:Kshow,kp,"$(colors[i_n])$(shapes[i_n])-",mec="k",label=latex("n=$n"),ms=8,lw=2)
    end
    title(latex("\\mathrm{$titlestring} (k_0=$k0)"),fontsize=17)
    xlim(0,Kshow)
    ylim(0,1.0)
    xticks(0:Kshow)
    xlabel(latex("k_{2\\%}  (\\mathrm{\\# of clusters with} >$(round(Int,100*cutoff))\\% \\mathrm{of points})"),fontsize=16)
    ylabel(latex("\\pi(k_{2\\%} | \\mathrm{data})"),fontsize=16)
    legend(fontsize=12)
end
plot_k_posteriors(4,1,"Standard posterior")
savefig("mixsim-kposterior-standard-k=$k0.png",dpi=150)
plot_k_posteriors(5,2,"Coarsened posterior")
savefig("mixsim-kposterior-coarsened-k=$k0.png",dpi=150)
plot_k_posteriors(6,3,"Conditional coarsening")
savefig("mixsim-kposterior-conditional-k=$k0.png",dpi=150)



nothing




