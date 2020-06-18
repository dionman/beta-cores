# Posterior c.d.f.s for simulation example with quadratic perturbation.
# __________________________________________________________________________________________
# Settings

ns = [50000]  # sample sizes to use
alphas = [Inf,NaN,1000,50]  # values of alpha to use
nreps = 5  # number of times to repeat each simulation
n_total = 10^4  # total number of MCMC samples
n_keep = n_total  # number of MCMC samples to record
n_burn = round(Int,n_keep/10)  # burn-in (of recorded samples)


# __________________________________________________________________________________________
# Helper functions

include("varsel.jl")
include("varsel-bayarri.jl")
include("skew.jl")

using VarSel, Bayarri
using PyPlot
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
beta0 = [-1,4,0,0,0,0]

function generate_sample(n)
    x = [ones(n) Skew.skewrndNormalized(n,Q,a)']
    x0 = vec(x[:,2])
    y = g(x0) + randn(n)*sigma
    return x,y
end

# Compute 100*P% interval from samples x
interval(x,P) = (N=length(x); xs=sort(x); ai=(1-P)/2; l=xs[floor(Int,ai*N)]; u=xs[ceil(Int,(1-ai)*N)]; (l,u))


# __________________________________________________________________________________________
# Run simulations

for fignum=1:length(alphas); figure(fignum,figsize=(6,8),dpi=70); clf(); draw(); end

nns = length(ns)
for (i_a,alpha) in enumerate(alphas)
    for (i_n,n) in enumerate(ns)
        for rep in 1:nreps
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
            end
                
            # Display c.d.f.s of coefficients
            xlims = Array[[-1.5,0.1],[2,4.5],[-.4,.4],[-.4,.4],[-.4,.4],[-.4,.4]]
            #figure(i_a+2,figsize=(8,8),dpi=70); clf()
            figure(i_a,figsize=(6,8),dpi=70); clf()
            subplots_adjust(bottom=0.1,hspace=0.4)
            for j=1:p
                subplot(p,1,j)
                betas = vec(beta_r[j,n_burn+1:end])
                n_use = n_keep - n_burn
                plot(sort([betas;betas]),sort([0:n_use-1;1:n_use])/n_use,"b-",markersize=0,linewidth=2)
                ylim(0,1)
                xlim(xlims[j])
                xm,xM = xlim(); ym,yM = ylim()
                text((xM-xm)*0.92+xm,(yM-ym)*0.5+ym,"\$\\beta_$j\$",fontsize=18)
                l,u = interval(betas,0.95)
                plot([l,l],ylim(),"r-",markersize=0)
                plot([u,u],ylim(),"r-",markersize=0)
                plot(beta0[j]*[1,1],ylim(),"k:",markersize=0)
            end
            subplot(p,1,1)
            ylabel("\$\\\mathrm{c.d.f.}\$",fontsize=16)
            if isnan(alpha); title(latex("\\mathrm{Mixture of }g\\mathrm{ priors}"),fontsize=17)
            elseif alpha<Inf; title("\$\\mathrm{Coarsened}\$",fontsize=17)
            else; title("\$\\mathrm{Standard}\$",fontsize=17)
            end
            #draw()
            savefig("varsel-beta-alpha=$alpha-n=$n-rep=$rep.png",dpi=150)
        end
    end
end




