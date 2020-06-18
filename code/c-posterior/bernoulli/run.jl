# Code for toy example involving Bernoulli trials
# ________________________________________________________________________________________________________
# Setup

# Settings
epsilon = 0.02 # a priori "precision" to use for coarsening
alpha = round(Int,1/(2*epsilon^2)) # a priori choice of alpha using relative entropy approach
theta_o = 0.51  # mean of observed data distribution
nns = 25 # 1+6*4 = number of n's to use
nreps = 10 # 1000 # number of times to run the simulation   <<<<< SET nreps=1000 TO REPRODUCE RESULTS IN PAPER
ns = round.(Int,logspace(0,6,nns)) # n's to use
from_file = false # load existing results from file

# H0: theta = 1/2
# H1: theta \neq 1/2, theta|H1 ~ Beta(1,1)

# ____________________________________________________________________________________________________________
# Helper functions

using PyPlot, Distributions
using JLD, HDF5
drawnow() = (pause(0.001); get_current_fig_manager()[:window][:raise_]())
logsumexp(x) = (m = maximum(x); m == -Inf ? -Inf : log(sum(exp.(x-m))) + m)
latex(s) = latexstring(replace(s," ","\\,\\,"))

# _______________________________________________________________________________________________________
# Calibrate alpha

if true
    srand(0) # reset RNG
    n = maximum(ns) # sample size to use for calibration
    y = (rand(n).<theta_o) # generate data
    s = sum(y)
    alphas = [10.^collect(1:0.1:5); Inf]  # values of alpha to consider
    n_alphas = length(alphas)
    
    # Compute the calibration curve
    Ek = zeros(n_alphas)  # posterior probability of H1, for each value of alpha
    El = zeros(n_alphas)  # posterior expectation of the log likelihood, for each value of alpha
    for (i_a,a) in enumerate(alphas)
        # power posterior approximation to the c-posterior
        alpha_n = 1/(1/a + 1/n)
        A = 1+alpha_n*s/n
        B = 1+alpha_n*(1-s/n)
        R10 = exp(alpha_n*log(2) + lbeta(A,B))
        r0x = 1/(1+R10)  # p(H0|E)
        Ek[i_a] = 1-r0x
        
        l0 = n*log(0.5)
        l1 = s*(digamma(A)-digamma(A+B)) + (n-s)*(digamma(B)-digamma(A+B))
        El[i_a] = r0x*l0 + (1-r0x)*l1
    end
    
    # Plot the calibration curve
    figure(1,figsize=(5.5,2.5)); clf() #; grid(true)
    subplots_adjust(left=0.15,bottom=0.2)
    plot(Ek,El,"ko-",ms=2,lw=1)
    if theta_o==0.5
        X,Y = Ek[end],El[end]
        alpha = Inf
        plot(X,Y,"rx",ms=10,mew=2)
        yl = ylim()
        text(X,Y+(yl[1]-yl[2])*0.2,latex("\\alpha=\\infty"),fontsize=14)
        ylim(yl[1],yl[2]+1500)
    elseif theta_o==0.51
        X,Y = Ek[25],El[25]
        alpha = 2500
        plot(X,Y,"rx",ms=10,mew=2)
        yl = ylim()
        text(X-0.01,Y-(yl[1]-yl[2])*0.10,latex("\\alpha=$(round(Int,alphas[25]))"),fontsize=14)
        X,Y = Ek[22],El[22]; plot(X,Y,"bo",ms=5,mew=2)
        ylim(yl[1],yl[2]+3000)
    elseif theta_o==0.75
        X,Y = Ek[end],El[end]
        alpha = Inf
        plot(X,Y,"rx",ms=10,mew=2)
        yl = ylim()
        text(X-0.09,Y+(yl[1]-yl[2])*0.05,latex("\\alpha=\\infty"),fontsize=14)
        ylim(yl[1],yl[2]+4000)
    else
        i_a = indmin(Ek)
        plot(Ek[i_a],El[i_a],"rx",ms=10,mew=2)
        alpha = alphas[i_a]
    end
    plot(Ek,El,"ko-",ms=2,lw=1)
    title(latex("\\mathrm{Calibration of} \\alpha  (\\theta^o=$theta_o)"),fontsize=17)
    ylabel(latex("\\mathrm{E}_\\alpha\\mathrm{(loglik | data)}"),fontsize=15)
    xlabel(latex("\\Pi_\\alpha(\\mathrm{H}_1 | \\mathrm{data})"),fontsize=16)
    gca()[:ticklabel_format](style="sci",axis="y",scilimits=(0,0),fontsize=8)
    yticks(fontsize=9)
    drawnow()
    savefig("bernoulli-calibration-theta_o=$theta_o.png",dpi=150)
end

# _______________________________________________________________________________________________________
# Compute results

# Initialize
p0xs = zeros(nreps,nns)
r0xs = zeros(nreps,nns)
e0xs = zeros(nreps,nns)
maxn = maximum(ns)

if from_file
    p0xs,r0xs,e0xs = load("bernoulli-theta_o=$theta_o-alpha=$alpha.jld","p0xs","r0xs","e0xs")
else
    for rep = 1:nreps  # for each run of the simulation
        println("rep = $rep")
        y = (rand(maxn).<theta_o) # generate data
        for (i,n) in enumerate(ns)  # for each value of n
            s = sum(y[1:n])  # sufficient statistic for first n data points

            # Standard posterior
            B10 = exp(n*log(2) + lbeta(1+s, 1+n-s)) # p(x|H1)/p(x|H0)
            p0x = 1/(1+B10)  # p(H0|x)
            
            # Coarsened posterior - Power posterior approximation
            alpha_n = 1/(1/alpha + 1/n)
            R10 = exp(alpha_n*log(2) + lbeta(1+alpha_n*s/n, 1+alpha_n*(1-s/n)))
            r0x = 1/(1+R10)  # p(H0|E)

            # Coarsened posterior - Exact calculation
            f(p,q) = (p==0? 0.0 : p*log(p/q))
            D(p,q) = f(p,q) + f(1-p,1-q) # relative entropy for Bernoulli
            t = collect(0:n)
            lpt0 = logpdf(Binomial(n,1/2),t) # Binomial(n,1/2)
            lpt1 = -log(n+1)*ones(n+1) # BetaBinomial(n,1,1) = Uniform{0,1,...,n}
            lpE0 = logsumexp(-alpha*D.(s/n,t/n) + lpt0)  # log(p(E|H0))
            lpE1 = logsumexp(-alpha*D.(s/n,t/n) + lpt1)  # log(p(E|H1))
            E10 = exp(lpE1 - lpE0)
            e0x = 1/(1+E10)  # p(H0|E)

            # record values
            p0xs[rep,i] = p0x
            r0xs[rep,i] = r0x
            e0xs[rep,i] = e0x

            #println("n = $n")
            #@printf("p(H0|x)  std=%.4f  approx=%.5f  exact=%.5f\n",p0x,r0x,e0x)
        end
    end
    save("bernoulli-theta_o=$theta_o-alpha=$alpha.jld","p0xs",p0xs,"r0xs",r0xs,"e0xs",e0xs)
end

# Plot results
figure(2,figsize=(5.5,2.5)); clf(); grid(true)
subplots_adjust(bottom = 0.22)
semilogx(ns,mean(p0xs,1)[:],label="standard posterior",linewidth=2,"gs-",markersize=5)
semilogx(ns,mean(e0xs,1)[:],label="exact c-posterior",linewidth=2,"r*-",markersize=10)
semilogx(ns,mean(r0xs,1)[:],label="approx c-posterior",linewidth=1,"bo-",markersize=5)
astr = (alpha<Inf? repr(round(Int,alpha)) : "\\infty")
title(latex("\\mathrm{Posterior}  (\\theta^o=$theta_o, \\alpha=$astr)"),fontsize=17)
xlabel(latex("n  \\mathrm{(sample size)}"),fontsize=16)
ylabel(latex("\\Pi(\\mathrm{H}_0 | \\mathrm{data})"),fontsize=15)
ylim(0,1)
#legend(numpoints=1,loc="upper right",fontsize=12,framealpha=1,borderaxespad=0.2)
#legend(numpoints=1,fontsize=12,framealpha=1,bbox_to_anchor=(0.7,0.6))
drawnow()
savefig("bernoulli-pH0-theta_o=$theta_o-alpha=$alpha.png",dpi=150)



nothing




