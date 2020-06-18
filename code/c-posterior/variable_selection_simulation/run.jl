# Simulation example for variable selection using power posterior.
# First example, involving a quadratic perturbation.
# __________________________________________________________________________________________
# Settings

ns = [100,1000,5000,10000,50000]  # sample sizes to use
alphas = [Inf,NaN,1000,50]  # values of alpha to use (NaN means to use Bayarri's mixture of g priors)
nreps = 10  # number of times to repeat each simulation
n_total = 5*10^4  # total number of MCMC samples
n_keep = n_total  # number of MCMC samples to record
n_burn = round(Int,n_keep/10)  # burn-in (of recorded samples)
from_file = false  # load previous results from file


# __________________________________________________________________________________________
# Helper functions

include("skew.jl")
include("varsel.jl")
include("varsel-bayarri.jl")

using Distributions, HDF5, JLD
using PyPlot
using VarSel, Bayarri
draw() = (pause(0.001); get_current_fig_manager()[:window][:raise_]())
latex(s) = latexstring(replace(s," ","\\,\\,"))


# __________________________________________________________________________________________
# Data

using Skew
a = [0.6,2.7,-3.3,-4.9,-2.5]
Q = [1.0 -0.89 0.93 -0.91 0.98
     -0.89 1.0 -0.94 0.97 -0.91
     0.93 -0.94 1.0 -0.96 0.97
     -0.91 0.97 -0.96 1.0 -0.93
     0.98 -0.91 0.97 -0.93 1.0]
sigma = 1
p = length(a)+1

g(x0) = -1 + 4*(x0 + (1/16)*x0.^2)

function generate_sample(n)
    x = [ones(n) Skew.skewrndNormalized(n,Q,a)']
    x0 = vec(x[:,2])
    y = g(x0) + randn(n)*sigma
    return x,y
end

# Compute histogram with the specified bin edges,
# where x[i] is in bin j if edges[j] < x[i] <= edges[j+1].
function histogram(x, edges=[]; n_bins=50, weights=ones(length(x)))
    if isempty(edges)
        mn,mx = minimum(x),maximum(x)
        r = mx-mn
        edges = linspace(mn-r/n_bins, mx+r/n_bins, n_bins+1)
    else
        n_bins = length(edges)-1
    end
    counts = zeros(Float64,n_bins)
    for i=1:length(x)
        for j=1:n_bins
            if (edges[j] < x[i] <= edges[j+1])
                counts[j] += weights[i]
                break
            end
        end
    end
    return counts,edges
end

# Plot data
srand(1)
figure(1,figsize=(5.5,3.2)); clf()
subplots_adjust(bottom=0.2)
n = 200
xmin,xmax = -4,4
ymin,ymax = -15,15
xs,ys = generate_sample(n)
xs0 = vec(xs[:,2])
plot(xs0,ys,"b.",markersize=3)
xt = linspace(xmin,xmax,5000)
plot(xt,g(xt),"k-")
xlim(xmin,xmax); ylim(ymin,ymax)
title(latex("\\mathrm{Data with quadratic perturbation}"),fontsize=17)
xlabel("\$x_{i 2}\$",fontsize=18)
ylabel("\$y_i\$",fontsize=18)
# draw()
savefig("varsel-data-n=$n.png",dpi=150)


# __________________________________________________________________________________________
# Calibration of alpha

if true
    alpha_candidates = [10.^collect(1:0.1:5); Inf]  # values of alpha to consider
    n_candidates = length(alpha_candidates)
    n = 10000  # sample size to use for calibration curve
    x,y = generate_sample(n)
    
    # Compute calibration curve
    if from_file
        El,Ek = load("varsel-calibration.jld","El","Ek")
    else
        El = zeros(n_candidates)  # posterior expectation of log likelihood, for each alpha
        Ek = zeros(n_candidates)  # posterior expectation of number of nonzero coefficients, for each alpha
        for (i_a,alpha) in enumerate(alpha_candidates)
            println("alpha=$alpha")
            zeta = (1/n)/(1/n + 1/alpha)
            tic()
            beta_r,lambda_r,keepers = VarSel.run(x,y,n_total,n_keep,zeta)
            toc()
            loglik_r = [0.5*n*log(lambda_r[ik]/(2*pi)) - 0.5*lambda_r[ik]*sum((y - x*beta_r[:,ik]).^2) for ik = 1:n_keep]
            El[i_a] = mean(loglik_r[n_burn+1:end])
            k_r = vec(sum(beta_r.!=0, 1))
            Ek[i_a] = mean(k_r[n_burn+1:end])
        end
        save("varsel-calibration.jld","El",El,"Ek",Ek)
    end

    # Plot calibration curve
    figure(2,figsize=(5.5,3.2)); clf()
    subplots_adjust(bottom=0.2)
    X,Y,alpha = Ek[21],El[21],alpha_candidates[21]
    plot(X,Y,"rx",ms=10,mew=2)
    plot(Ek,El,"ko-",ms=2,lw=1)
    yl = ylim()
    ylim(yl[1],yl[2]+800)
    xlim(1.8,xlim()[2])
    text(X-0.07,Y-(yl[1]-yl[2])*0.07,latex("\\alpha=$(round(Int,alpha))"),fontsize=14)
    X,Y,alpha = Ek[8],El[8],alpha_candidates[8]
    plot(X,Y,"bo",ms=5,mew=2)
    text(X-0.33,Y+(yl[1]-yl[2])*0.12,latex("\\alpha=$(round(Int,alpha))"),fontsize=14)
    gca()[:ticklabel_format](style="sci",axis="y",scilimits=(0,0),fontsize=8)
    yticks(fontsize=9)
    title(latex("\\mathrm{Calibration of} \\alpha"),fontsize=17)
    ylabel(latex("\\hat\\mathrm{E}_\\alpha\\mathrm{(loglik | data)}"),fontsize=15)
    xlabel(latex("\\hat\\mathrm{E}_\\alpha(k | \\mathrm{data})"),fontsize=16,labelpad=0)
    draw()
    savefig("varsel-calibration.png",dpi=150)
end


# __________________________________________________________________________________________
# Run simulations

nns = length(ns)
for (i_a,alpha) in enumerate(alphas)
    if from_file
        k_posteriors = load("k_posteriors-alpha=$alpha.jld","k_posteriors")
    else
        k_posteriors = zeros(p+1,nreps,nns)
        for (i_n,n) in enumerate(ns)
            for rep in 1:nreps
                println("alpha=$alpha   n=$n   rep=$rep")
                srand(n+rep) # Reset RNG
                
                # Sample data
                x,y = generate_sample(n)
                x0 = vec(x[:,2])

                # Run sampler
                if !isnan(alpha)
                    zeta = (1/n)/(1/n + 1/alpha)
                    beta_r,lambda_r,keepers = VarSel.run(x,y,n_total,n_keep,zeta)
                else
                    beta_r,lambda_r,g_r,keepers = Bayarri.run(x,y,n_total,n_keep)

                    # traceplot of g
                    figure(3); clf()
                    plot(g_r + 0.5*rand(length(g_r))-0.25,"k.",markersize=2)
                    # draw()
                    # savefig("traceplot-g-alpha=$alpha-n=$n-rep=$rep.png",dpi=150)
                end
                k_r = vec(sum(beta_r.!=0, 1))

                # Compute posterior on k
                counts,edges = histogram(k_r[n_burn+1:end],-0.5:1:p+0.5)
                k_posteriors[:,rep,i_n] = counts/(n_keep-n_burn)

                # traceplot of number of nonzero coefficients
                figure(4); clf()
                plot(k_r + 0.5*rand(length(k_r))-0.25,"k.",markersize=2)
                # draw()
                # savefig("traceplot-alpha=$alpha-n=$n-rep=$rep.png",dpi=150)
            end
        end
        save("k_posteriors-alpha=$alpha.jld","k_posteriors",k_posteriors)
    end
    
    # Plot results
    figure(10+i_a,figsize=(5.5,3.2)); clf()
    subplots_adjust(bottom=0.2)
    colors = "bgyrm"
    shapes = "ds^v*"
    plot(0:p,VarSel.prior_k.(0:p,p),"ko-",label=latex("\\mathrm{prior}"),markersize=8,linewidth=2)
    for (i_n,n) in enumerate(ns)
        plot(0:p,vec(mean(k_posteriors[:,:,i_n],2)),"$(colors[i_n])$(shapes[i_n])-",label=latex("n=$n"),mec="k",ms=8,lw=2)
    end
    xlim(0,p)
    ylim(0,1)
    xticks(0:p)
    if isnan(alpha); title(latex("\\mathrm{Mixture of }g\\mathrm{ priors}"),fontsize=17)
    elseif alpha<Inf; title(latex("\\mathrm{Coarsened}  (\\alpha=$(round(Int,alpha)))"),fontsize=17)
    else; title(latex("\\mathrm{Standard}"),fontsize=17)
    end
    xlabel(latex("k  \\mathrm{(\\# of nonzero coefficients)}"),fontsize=17)
    ylabel(latex("\\pi(k|\\mathrm{data})"),fontsize=13)
    #legend(numpoints=1,bbox_to_anchor=(1.02, 0.9), loc=2, borderaxespad=0.)
    if alpha<Inf
        legend(numpoints=1,loc="upper right",framealpha=0.8,fontsize=15,labelspacing=0.2,borderaxespad=0.1)
    end
    #draw()
    savefig("varsel-kposteriors-alpha=$alpha.png",dpi=150)
end



