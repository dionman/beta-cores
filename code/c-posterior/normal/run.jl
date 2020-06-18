# Toy normal example with outliers to demonstrate using Wasserstein distance in a coarsened posterior.

using Distributions
using PyPlot
using JLD, HDF5

include("helper.jl")

# ________________________________________________________________________________________________________
# Settings

ns = [50,200,1000,10000]  # sample sizes to use
perturbed = true  # use perturbed distribution to generate the data, or not (if true, the model is misspecified)
from_file = false  # use previous results from saved file
p_outlier = 0.1  # mixture weight of outlier component
m_outlier = 20  # mean of outlier component
n_iterations = 200000  # total number of MCMC iterations
n_burn = round(Int,n_iterations/10)  # number of MCMC iterations to discard as burn-in

# Prior
m_m,m_s = 0,5  # mean and standard deviation of the prior on m
m_prior = Normal(m_m,m_s)  # prior on the mean m
v_a,v_b = 1,1  # parameters of the prior on v
v_prior = InverseGamma(v_a,v_b)  # prior on the variance v
a_a,b_a = 7,0  # parameters of the hyperprior on the coarsening parameter a (alpha)
# Note that this is an improper prior when b_a=0.  It's an unusual choice when a_a>1 since it diverges as a->infty, but it still yields a proper posterior, and it appears to yield consistency as n->infty.
a_prior = Gamma(a_a,1/b_a)   # prior on the coarsening parameter a (alpha)
z_prior = Normal(0,1)  # distribution of the latent variables z_1,...,z_n

# Proposal scales
sigma_mp = 0.25  # standard deviation for the Metropolis proposals on m
sigma_vp = 0.25  # standard deviation for the Metropolis proposals on v
sigma_zp = 0.02  # standard deviation for the Metropolis proposals on z

# Define the distance as a function of xo=sort(x) and zo,m,v (where zo=sort(z))
d(xo,zo,m,v) = mean(abs.(zo*sqrt(v)+m - xo))  # 1st Wasserstein distance between x and z*sqrt(v)+m


# ________________________________________________________________________________________________________
# Generating simulated data 

# True (idealized) distribution: x_i ~ N(m0,v0)
m0 = 3.2  # idealized mean
v0 = 4.4  # idealized variance

# Using a Dirichlet process mixture model, randomly perturb the true (idealized) distribution to create an observed data distribution.
srand(1)  # reset the random number generator
L = 10000  # number of components for DP approximation
alpha0 = 500  # DP concentration parameter
s_t = [1.0; 0.25*ones(L)]  # component standard deviations
beta_t = randn(L)*sqrt(v0) + m0
beta_t = [m_outlier; beta_t]  # component means
p_t = rand(Dirichlet(L,alpha0/L))
p_t = [p_outlier; (1-p_outlier)*p_t]  # component weights

# Function to generate simulated data from the observed data distribution
function generate_data(n)
    if perturbed
        z_t = rand(Categorical(p_t),n)
        x = randn(n).*s_t[z_t] + beta_t[z_t]
    else
        x = randn(n)*sqrt(v0) + m0
    end
    return x
end

# Generate and plot simulated data
if true
    n = 10000 # 100000
    xmin,xmax = -6,24
    x = generate_data(n)
    
    figure(20,figsize=(8,2.5)); clf()
    subplots_adjust(bottom=0.2)
    xs = linspace(xmin,xmax,1000)
    pertlabel = (perturbed? "perturbed" : "unperturbed")
    plot_histogram(x, titlestring=latex("\\mathrm{Data distribution ($pertlabel)}"),edges=linspace(xmin,xmax,2*sqrt(n)),color="#aaaaaa",edgecolor="#cccccc",linewidth=0.1)
    plot(xs,pdf(Normal(m0,sqrt(v0)),xs),"b--",lw=2)
    if perturbed
        density = Float64[sum(p_t.*normpdf(xi,beta_t,s_t)) for xi in xs]
        plot(xs,density,"r-",lw=1.5)
    end
    xlabel(latex("x"),fontsize=16)
    ylabel(latex("\\mathrm{density}"),fontsize=16)
    xlim(xmin,xmax)
    ymax = ylim()[2]
    xticks(xmin:2:xmax)
    drawnow()
    savefig("normal-data-perturbed=$perturbed-n=$n.png",dpi=150)
end


# ________________________________________________________________________________________________________
# Define MCMC algorithms

function run_coarsened_sampler(x,n_iterations)
    xo = sort(x)
    n = length(x)

    # Initialize
    m = 0.0
    v = 1.0
    a = 1.0
    zo = sort(rand(z_prior,n))
    ms = zeros(n_iterations)
    vs = zeros(n_iterations)
    as = zeros(n_iterations)
    n_accept = 0

    # Run
    for iteration = 1:n_iterations
        # Update a
        a = rand(Gamma(a_a, 1/(b_a + d(xo,zo,m,v))))

        # Update z
        zp = sort(zo + sigma_zp*randn(n))
        llp = sum(logpdf(z_prior,zp)) - a*d(xo,zp,m,v)
        ll  = sum(logpdf(z_prior,zo)) - a*d(xo,zo,m,v)
        if rand() < exp(llp-ll); zo = zp; n_accept += 1; end
        
        # Update m and v
        mp = m + randn()*sigma_mp
        vp = v + randn()*sigma_vp
        llp = (vp>0? logpdf(m_prior,mp) + logpdf(v_prior,vp) - a*d(xo,zo,mp,vp) : -Inf)
        ll  = logpdf(m_prior,m) + logpdf(v_prior,v) - a*d(xo,zo,m,v)
        if rand() < exp(llp-ll); m = mp; v = vp; end
        
        # Record-keeping
        ms[iteration] = m
        vs[iteration] = v
        as[iteration] = a
    end
    @printf "Acceptance rate for z: %.6f%%\n" 100*n_accept/n_iterations

    return ms,vs,as,zo
end

function run_standard_sampler(x,n_iterations)
    n = length(x)

    # Initialize
    m = 0.0
    v = 1.0
    ms = zeros(n_iterations)
    vs = zeros(n_iterations)
    
    # Precompute some stuff
    l0 = 1/m_s^2
    sum_x = sum(x)
    sum_xx = sum(x.*x)

    # Run
    for iteration = 1:n_iterations
        # Update m
        l = 1/v
        lp = l0 + n*l
        mp = (l0*m_m + l*sum_x) / lp
        m = randn()/sqrt(lp) + mp
        
        # Update v
        ap = v_a + 0.5*n
        bp = v_b + 0.5*(sum_xx - 2*m*sum_x + n*m*m)
        v = rand(InverseGamma(ap,bp))
        
        # Record-keeping
        ms[iteration] = m
        vs[iteration] = v
    end
    return ms,vs
end


# ________________________________________________________________________________________________________
# Run algorithms

# Generate data
srand(3)  # choose a representative example
x_all = generate_data(maximum(ns))

if from_file
    (mr,vr,ar) = load("results-perturbed=$perturbed.jld","results")
else
    nns = length(ns)
    mr = Array{Array{Float64,1},2}(nns,2)
    vr = Array{Array{Float64,1},2}(nns,2)
    ar = Array{Array{Float64,1},2}(nns,2)

    for (i_n,n) in enumerate(ns)
        println("n = $n")
        x = x_all[1:n]
        
        # Run MCMC sampler for standard posterior
        println("Running standard posterior sampler...")
        @time ms,vs = run_standard_sampler(x,n_iterations)
        mr[i_n,1] = ms; vr[i_n,1] = vs; ar[i_n,1] = Inf*ones(n_iterations)
        
        # Run MCMC sampler for coarsened posterior
        println("Running coarsened posterior sampler...")
        @time ms,vs,as,zo = run_coarsened_sampler(x,n_iterations)
        mr[i_n,2] = ms; vr[i_n,2] = vs; ar[i_n,2] = as
    end

    # Save to file
    save("results-perturbed=$perturbed.jld","results",(mr,vr,ar))
end

# ________________________________________________________________________________________________________
# Plot results

use = n_burn+1:n_iterations
n_use = length(use)
fignum = 0

# Make traceplots of m and v and a to assess MCMC convergence
if false
    for (i_n,n) in enumerate(ns)
        for (i_t,t) in enumerate(["standard","coarsened"])
            close(50); figure(50,figsize=(8,2.5)); clf(); subplots_adjust(bottom=0.2)
            plot(mr[i_n,i_t],"k.",ms=0.1); title(latex("\\mathrm{Traceplot of} \\mu")); drawnow(); savefig("normal-traceplot-m-$t-n=$n.png",dpi=150)
            close(50); figure(50,figsize=(8,2.5)); clf(); subplots_adjust(bottom=0.2)
            plot(vr[i_n,i_t],"k.",ms=0.1); title(latex("\\mathrm{Traceplot of} \\sigma^2")); drawnow(); savefig("normal-traceplot-v-$t-n=$n.png",dpi=150)
            close(50); figure(50,figsize=(8,2.5)); clf(); subplots_adjust(bottom=0.2)
            plot(ar[i_n,i_t],"k.",ms=0.1); title(latex("\\mathrm{Traceplot of} \\alpha")); drawnow(); savefig("normal-traceplot-a-$t-n=$n.png",dpi=150)
        end
    end
end

# Plot posterior of m
for (i_t,t) in enumerate(["Standard","Coarsened"])
    figure(fignum+=1,figsize=(8,2.5)); clf(); subplots_adjust(bottom=0.2)
    xmin,xmax,step = (perturbed? (0,10,1) : (2.5,4.0,0.2))
    for (i_n,n) in enumerate(ns)
        plot([xmin;sort(mr[i_n,i_t][use]);xmax],[0;(1:n_use)/n_use;1],color=T10colors[i_n],"-",lw=2,label=latex("n=$n"))  # plot CDF
    end
    plot([m0,m0],ylim(),"k--",linewidth=2)
    xlim(xmin,xmax)
    xticks(xmin:step:xmax)
    ylim(0,1)
    pertlabel = (perturbed? "perturbed" : "unperturbed")
    title(latex("\\mathrm{$t posterior c.d.f. of} \\mu  \\mathrm{($pertlabel)}"),fontsize=17)
    legend(fontsize=14,loc="lower right")
    xlabel(L"\mu",fontsize=15)
    ylabel(L"\mathrm{c.d.f.}",fontsize=14)
    drawnow()
    savefig("normal-cdf-m-perturbed=$perturbed-$t.png",dpi=150)
end

# Plot posterior of v
for (i_t,t) in enumerate(["Standard","Coarsened"])
    figure(fignum+=1,figsize=(8,2.5)); clf(); subplots_adjust(bottom=0.2)
    xmin,xmax,step = (perturbed? (0,50,5) : (1,7,0.5))
    for (i_n,n) in enumerate(ns)
        plot([xmin;sort(vr[i_n,i_t][use]);xmax],[0;(1:n_use)/n_use;1],color=T10colors[i_n],"-",lw=2,label=latex("n=$n"))  # plot CDF
    end
    plot([v0,v0],ylim(),"k--",linewidth=2)
    xlim(xmin,xmax)
    xticks(xmin:step:xmax)
    ylim(0,1)
    pertlabel = (perturbed? "perturbed" : "unperturbed")
    title(latex("\\mathrm{$t posterior c.d.f. of} \\sigma^2  \\mathrm{($pertlabel)}"),fontsize=17)
    legend(fontsize=14,loc="lower right")
    xlabel(L"\sigma^2",fontsize=15)
    ylabel(L"\mathrm{c.d.f.}",fontsize=14)
    drawnow()
    savefig("normal-cdf-v-perturbed=$perturbed-$t.png",dpi=150)
end

# Plot posterior of a
figure(fignum+=1,figsize=(8,2.5)); clf(); subplots_adjust(bottom=0.2)
xmin,xmax = (perturbed? (0,15) : (0,700))
for (i_n,n) in enumerate(ns)
    plot([xmin;sort(ar[i_n,2][use]);xmax],[0;(1:n_use)/n_use;1],color=T10colors[i_n],"-",lw=2,label=latex("n=$n"))  # plot CDF
end
xlim(xmin,xmax)
ylim(0,1)
pertlabel = (perturbed? "perturbed" : "unperturbed")
title(latex("\\mathrm{Coarsened posterior c.d.f. of} \\alpha  \\mathrm{($pertlabel)}"),fontsize=17)
legend(fontsize=14,loc="lower right")
xlabel(L"\alpha",fontsize=15)
ylabel(L"\mathrm{c.d.f.}",fontsize=14)
drawnow()
savefig("normal-cdf-a-perturbed=$perturbed.png",dpi=150)






