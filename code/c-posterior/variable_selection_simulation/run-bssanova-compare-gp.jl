# Simulation example for variable selection using power likelihood.
# Compare with BSSANOVA on second example, involving Gaussian process perturbation.
# __________________________________________________________________________________________
# Settings

ns = [100,1000,5000]  # sample sizes to use
alphas = [Inf,50]  # values of alpha to use
nreps = 1  # number of times to repeat each simulation
n_total = 5*10^4  # total number of MCMC samples
n_keep = n_total  # number of MCMC samples to record
n_burn = round(Int,n_keep/10)  # burn-in (of recorded samples)

# __________________________________________________________________________________________
# Helper functions

include("varsel.jl")
include("skew.jl")

using Distributions, HDF5, JLD
using PyPlot
using VarSel
draw() = (pause(0.001); get_current_fig_manager()[:window][:raise_]())
counts(a,k) = (n=zeros(Int,k); for ai in a; n[ai] += 1; end; n)

# Compute 100*P% interval from samples x
interval(x,P) = (N=length(x); xs=sort(x); ai=(1-P)/2; l=xs[floor(Int,ai*N)]; u=xs[ceil(Int,(1-ai)*N)]; (l,u))

# __________________________________________________________________________________________
# Data

# Gaussian process
function GP(z,sigma,lengthscale)
    D = abs.(z .- z')
    lambda = 1/lengthscale^2
    C = (sigma^2)*exp.(-0.5*lambda*D.*D)
    U,S,V = svd(C)
    A = U*diagm(sqrt.(S))*V'
    return A*randn(length(z))
end

# Linearly interpolate y=f(x) to xi
function interpolate(x,y,xi)
    n = length(x)
    ni = length(xi)
    order = sortperm(x)
    xs = x[order]
    ys = y[order]
    a = diff(ys)./diff(xs)
    b = ys[1:end-1]
    orderi = sortperm(xi)
    yi = zeros(ni)
    j = 1
    for i in orderi
        while (xs[j+1]<xi[i]) && (j+1<n); j+=1; end
        yi[i] = a[j]*(xi[i]-xs[j]) + b[j]
    end
    return yi
end

# Data distribution
using Skew
a = [0.6,2.7,-3.3,-4.9,-2.5]
Q = [1.0 -0.89 0.93 -0.91 0.98
     -0.89 1.0 -0.94 0.97 -0.91
     0.93 -0.94 1.0 -0.96 0.97
     -0.91 0.97 -0.96 1.0 -0.93
     0.98 -0.91 0.97 -0.93 1.0]
sigma = 1
p = length(a)+1

srand(1)
beta0 = [-1,4,0,0,0,0]
f0(x) = vec(x*beta0)
xerr = linspace(-5,5,500)
serr = [[0,1];.25*ones(4)]
err = zeros(500,p); for j=2:p; err[:,j] = GP(xerr,serr[j],1); end
fc(x) = (y=f0(x); for j=2:p; y += interpolate(xerr,err[:,j],x[:,j]); end; y)  # perturbed regression function

function generate_sample(n)
    x = [ones(n) Skew.skewrndNormalized(n,Q,a)']
    y = fc(x) + randn(n)*sigma
    return x,y
end

xmin,xmax = -4,4
nt = 5000
xt = [ones(nt) linspace(xmin,xmax,nt) zeros(nt,4)]

figure(1,figsize=(6.5,3.2)); clf()
subplots_adjust(bottom=0.2,right=0.75)
plot(xt[:,2],f0(xt)-f0(xt),"k--",label=L"\mathrm{True}")
#plot(xt[:,2],fc(xt)-f0(xt),"k-",label=L"\mathrm{Perturbed}",lw=1)
n = 5000
nshow = 200
srand(n+1) # set RNG to match simulation used below for plotting
x,y = generate_sample(n)
order = sortperm(x[1:nshow,2])
plot(x[order,2],fc(x[order,:])-f0(x[order,:]),"ko",ms=4,mfc="none",label=L"\mathrm{Perturbed}")


# __________________________________________________________________________________________
# Run simulations

nns = length(ns)
timepersample = zeros(length(alphas),nns)
for (i_a,alpha) in enumerate(alphas)
    k_posteriors = zeros(p+1,nreps,nns)
    for (i_n,n) in enumerate(ns)
        for rep in 1:nreps
            srand(n+rep) # Reset RNG
            
            # Sample data
            x,y = generate_sample(n)

            # Run sampler
            zeta = (1/n)/(1/n + 1/alpha)
            tic()
            beta_r,lambda_r,keepers = VarSel.run(x,y,n_total,n_keep,zeta)
            timepersample[i_a,i_n] = toq()/n_total
            println("Time per sample: ",timepersample[i_a,i_n] )
            k_r = vec(sum(beta_r.!=0, 1))

            # Compute posterior on k
            k_posteriors[:,rep,i_n] = counts(k_r[n_burn+1:end]+1,p+1)/(n_keep-n_burn)

            # Compute intervals for regression function
            if n==5000 && rep==1
                figure(1)
                lower,upper = zeros(nshow),zeros(nshow)
                mu,sd = zeros(nshow),zeros(nshow)
                for i = 1:nshow
                    fs = beta_r'*x[i,:]
                    lower[i],upper[i] = interval(fs[n_burn+1:end],0.99)  # 99% interval for regression function values
                    mu[i] = mean(fs[n_burn+1:end])
                    sd[i] = std(fs[n_burn+1:end])
                end
                xshow = x[1:nshow,:]
                mtrue = xshow*beta0
                println("Fraction of 99% intervals containing true value: ", mean(lower.<mtrue.<upper))
                mark = (alpha<Inf? "x" : "+")
                col = (alpha<Inf? "b" : "g")
                lab = (alpha<Inf? L"\mathrm{Coarsened}" : L"\mathrm{Standard}")
                mult = quantile(Normal(0,1),0.995)
                plot(xshow[:,2],mu+mult*sd-mtrue,col*mark,markersize=4,label=lab) # use sd for most direct comparison with BSSANOVA
                plot(xshow[:,2],mu-mult*sd-mtrue,col*mark,markersize=4)
                # plot(xshow[:,2],fc(xshow)-f0(xshow),"ko",ms=2)
                draw()
            end
            #writedlm("gp-bssanova-data-n=$n.dat",[y x[:,2:end]])
        end
    end
    
    # Plot results
    figure(10+i_a,figsize=(5.5,3.2)); clf()
    subplots_adjust(bottom=0.2,right=0.75)
    colors = "bgyrm"
    shapes = "ds^v*"
    #plot(0:p,VarSel.prior_k(0:p,p),"ko-",label=L"\mathrm{prior}",markersize=7,linewidth=2)
    for (i_n,n) in enumerate(ns)
        plot(0:p,vec(mean(k_posteriors[:,:,i_n],2)),"$(colors[i_n])$(shapes[i_n])-",label="\$n = $n\$",markersize=7,linewidth=2,mec="k")
    end
    xlim(0,p)
    ylim(0,1.0)
    xticks(0:p)
    if alpha<Inf; title(L"\mathrm{Coarsened\,\,\,(GP\,\,perturbation)}",fontsize=17)
    else; title(L"\mathrm{Standard\,\,\,(GP\,\,perturbation)}",fontsize=17)
    end
    xlabel(L"k \,\,\,\mathrm{(\#\,\,of\,\,nonzero\,\,coefficients)}",fontsize=16)
    ylabel(L"\pi(k|\mathrm{data})",fontsize=15, labelpad=0)
    #legend(numpoints=1,bbox_to_anchor=(1.02, 0.9), loc=2, borderaxespad=0., fontsize=15)
    legend(numpoints=1, loc="upper right", fontsize=15)
    draw()
    savefig("varsel-gp-kposteriors-alpha=$alpha.png",dpi=150)
end
println("Time per sample:")
display(timepersample)


# __________________________________________________________________________________________
# Plot BSSANOVA results

# Intervals for regression function
figure(1)
n = 5000
rep = 1
D = readdlm("bssanova-output/gp-bssanova-n=$n-fit.dat")
ints = readdlm("bssanova-output/gp-bssanova-n=$n-int.dat")
curves = readdlm("bssanova-output/gp-bssanova-n=$n-curves.dat")
curvessd = readdlm("bssanova-output/gp-bssanova-n=$n-curvessd.dat")
mu = D[1:nshow,1]
sd = D[1:nshow,2]
srand(n+rep) # Reset RNG
x,y = generate_sample(n)
mtrue = x*beta0
mult = quantile(Normal(0,1),0.995)
println("BSSANOVA: Fraction of 99% intervals containing true value: ", mean(mu-mult*sd.<mtrue[1:nshow].<mu+mult*sd))
plot(x[1:nshow,2],mu+mult*sd-mtrue[1:nshow],"r^",markersize=2,label=L"\mathrm{BSSANOVA}")
plot(x[1:nshow,2],mu-mult*sd-mtrue[1:nshow],"r^",markersize=2)
order = sortperm(x[:,2])
yl = ylim()
#plot(x[order,2],mean(ints)+curves[order,1]+mult*curvessd[order,1]-mtrue[order],"r-",ms=0,lw=0.5)
#plot(x[order,2],mean(ints)+curves[order,1]-mult*curvessd[order,1]-mtrue[order],"r-",ms=0,lw=0.5)
title(L"\mathrm{Credible\,\,intervals\,\,\,(GP\,\,perturbation)}",fontsize=17)
xlabel(L"x_{i 2}",fontsize=16)
ylabel(L"f(x) - f_{\theta_I}(x)",fontsize=15, labelpad=-1)
legend(numpoints=1,bbox_to_anchor=(1.02, 0.9), loc=2, borderaxespad=0., markerscale=2, fontsize=14, handlelength=1, handletextpad=0.3)
xlim(-2.7,2.7)
ylim(yl[1],yl[2])
draw()
savefig("varsel-gp-bssanova-curves.png",dpi=150)
# 99\% credible intervals for the (true) regression function $f(x)$ at the first 500 points $x$ from the data set; the projection onto the second covariate is shown.

# Standard: Fraction of 99% intervals containing true value: 0.125
# Coarsened: Fraction of 99% intervals containing true value: 1.0
# BSSANOVA: Fraction of 99% intervals containing true value: 0.26

# Posterior on k
figure(2,figsize=(5.5,3.2)); clf()
subplots_adjust(bottom=0.2,right=0.75)
colors = "bgyrm"
shapes = "ds^v*"
#plot(0:p,[0;pdf(Binomial(p-1,0.5),0:p-1)],"ko-",label=L"\mathrm{prior}",markersize=7,linewidth=2)
for (i_n,n) in enumerate([100,1000,5000])
    k_posterior = readdlm("bssanova-output/gp-bssanova-n=$n-kposterior.dat")
    plot(0:p,[0;vec(k_posterior)],"$(colors[i_n])$(shapes[i_n])-",label="\$n = $n\$",markersize=7,linewidth=2,mec="k")
end
xlim(0,p)
xticks(0:p)
ylim(0,1)
title(L"\mathrm{BSSANOVA\,\,\,(GP\,\,perturbation)}",fontsize=17)
xlabel(L"k \,\,\,\,\mathrm{(\#\,\,of\,\,nonzero\,\,coefficients)}",fontsize=17)
ylabel(L"\pi(k|\mathrm{data})",fontsize=15, labelpad=0)
#legend(numpoints=1,bbox_to_anchor=(1.02, 0.9), loc=2, borderaxespad=0.,fontsize=15)
#legend(numpoints=1, loc="upper right", fontsize=15)
draw()
savefig("varsel-gp-bssanova-kposteriors.png",dpi=150)




