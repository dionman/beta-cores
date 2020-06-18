# Apply variable selection using power likelihood to analyze the effect of various exposures
# on birth weight, using data from the Collaborative Perinatal Project (CPP).

include("varsel.jl")
using VarSel
using PyPlot
using JLD, HDF5

# ____________________________________________________________________________________________________________________________
# Settings

n_total = 2*10^4  # total number of MCMC samples
n_keep = n_total  # number of MCMC samples to record
n_burn = round(Int,n_keep/10)  # burn-in (of recorded samples)
alphas_for_kplot = [250,800,1500,2500,Inf]  # values of alpha to use for posterior on k, etc.
alphas_for_curve = [round.(10.^collect(1:0.1:5),1); Inf]  # values of alpha to use for calibration curve, etc.
alphas = sort(unique([alphas_for_kplot;alphas_for_curve]))
exclude_mwgt = true  # exclude mother's weight as a covariate
from_file = false  # load existing results from file


# ____________________________________________________________________________________________________________________________
# Data

# Variables to use
target_name = "V_BWGT"  # name of target variable
control_names = vec(readdlm("CPP-data/controls.tsv")[:,1])  # names of the non-exposure control variables
if exclude_mwgt; control_names = control_names[control_names.!="V_MWGTPP"]; end
exposure_names = vec(readdlm("CPP-data/exposures.tsv")[:,1])  # names of the exposure variables
predictor_names = [control_names; exposure_names]

# randomly permute order to check for possible MCMC mixing issues
#predictor_names = predictor_names[randperm(length(predictor_names))]

# Load data
values,names = readdlm("CPP-data/data.csv",','; header=true)
predictor_subset = [findfirst(names.==name)::Int64 for name in predictor_names]
@assert(minimum(predictor_subset)>0)
target_index = findfirst(names.==target_name)
n_missing_for_each_var = vec(sum(values[:,predictor_subset].==".", 1))
for j in predictor_subset
    missing = vec(values[:,j].==".")
    values[missing,j] = mean(values[.!missing,j])
end
println("Missing values in covariates have been replaced by the average of non-missing entries.")
missing_y = vec(values[:,target_index].==".")
y = convert(Array{Float64,1}, values[.!missing_y,target_index])
x = convert(Array{Float64,2}, values[.!missing_y,predictor_subset])
n = length(y)
println("Removed $(sum(missing_y)) records that were missing the target variable.")

# Preprocess data
y = (y-mean(y))./std(y)
x = (x.-mean(x,1))./std(x,1)
@assert(all(.!isnan.(y)))
@assert(all(.!isnan.(x)))
x = [ones(n) x] # append constant for intercept
predictor_names = ["CONST";predictor_names]
p = size(x,2)


# ____________________________________________________________________________________________________________________________
# Miscellaneous

latex(s) = latexstring(replace(s," ","\\,\\,"))
draw() = (pause(0.001); get_current_fig_manager()[:window][:raise_]())
# Color palette from https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
colors = ["#e6194b", "#3cb44b", "#0082c8", "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#d2f53c", "#fabebe", "#008080", "#e6beff", "#aa6e28", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000080", "#808080", "#ffe119", "#FFFFFF", "#000000"]
colors[21] = "#07c6b3"  # replace white with turquoise
T10colors = collect("bgyrmck"^100)
shapes = collect("ds^v*o."^100)

function interval(x,P)
    N = length(x)
    xsort = sort(x)
    ai = (1-P)/2
    l,u = xsort[round(Int,floor(ai*N))], xsort[round(Int,ceil((1-ai)*N))]
    return l,u
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


# ____________________________________________________________________________________________________________________________
# Run MCMC algorithm

# Initialize
n_alphas = length(alphas)
use = n_burn+1:n_keep
n_use = n_keep - n_burn
k_posteriors = zeros(p+1,n_alphas)
P_posteriors = zeros(p,n_alphas)
beta_means = zeros(p,n_alphas)
El = zeros(n_alphas)
Ek = zeros(n_alphas)

# Run
if from_file
    (alphas,k_posteriors,P_posteriors,beta_means,El,Ek)=load("cpp-results-"*(exclude_mwgt?"wo":"w")*"-mwgt.jld","results")
else
    for (i_a,alpha) in enumerate(alphas)
        srand(1)
        println("alpha = $alpha")

        # Run sampler
        zeta = (1/n)/(1/n + 1/alpha)
        tic()
        beta_r,lambda_r,loglik_r,keepers = VarSel.run(x,y,n_total,n_keep,zeta)
        toc()
        k_r = vec(sum(beta_r.!=0, 1))
        P_nonzero = vec(mean(beta_r[:,use].!=0, 2))
        P_posteriors[:,i_a] = P_nonzero
        beta_means[:,i_a] = vec(mean(beta_r[:,use],2))
        El[i_a] = mean(loglik_r[use])
        Ek[i_a] = mean(k_r[use])

        # Compute posterior on k
        counts,edges = histogram(k_r[use],-0.5:1:p+0.5)
        k_posteriors[:,i_a] = counts/n_use

        # Show traceplot of number of nonzero coefficients
        figure(1); clf()
        plot(k_r + 0.5*rand(length(k_r))-0.25,"k.",markersize=2)
        #draw()
        #savefig("cpp-traceplot-k-alpha=$alpha.png",dpi=150)
        
        # Show posterior of lambda
        figure(2); clf()
        counts,edges = histogram(lambda_r[use])
        plot((edges[1:end-1]+edges[2:end])/2,counts)
        #draw()
        #savefig("cpp-lambda-alpha=$alpha.png",dpi=150)
        #println("E(lambda|data) = ", mean(lambda_r[use]))

        # List of most probable nonzero coefs
        order = sortperm(P_nonzero,rev=true)

        # Show traceplots of top coefficients
        figure(3,figsize=(8,8)); clf()
        subplots_adjust(bottom=0.1,hspace=0.4)
        for ji = 1:8
            j = order[ji]
            subplot(8,1,ji)
            betas = vec(beta_r[j,:])
            plot(betas,"k.",markersize=2)
            xm,xM = xlim(); ym,yM = ylim()
            text((xM-xm)*0.05+xm,(yM-ym)*0.5+ym,predictor_names[j])
        end
        #draw()
        #savefig("cpp-traceplot-beta-alpha=$alpha.png",dpi=150)
        
        # Display c.d.f.s of top coefficients
        figure(4,figsize=(5.0,6)); clf()
        #subplots_adjust(bottom=0.1,hspace=0.5,wspace=0.02)
        subplots_adjust(bottom=0.1,hspace=0.05,wspace=0.02)
        for ji = 1:10 #15
            j = order[ji]
            subplot(5,2,ji)
            betas = vec(beta_r[j,use])
            plot(sort([betas;betas]),sort([(0:n_use-1); (1:n_use)])/n_use,"b-",markersize=0,linewidth=2)
            ylim(0,1)
            #mb = max(0.01, maximum(abs.(betas))); xlim(-mb,mb)
            xlim(-0.3,0.3)
            xm,xM = xlim(); ym,yM = ylim()
            #text((xM-xm)*0.05+xm,(yM-ym)*0.5+ym,predictor_names[j])
            #title(predictor_names[j],fontsize=11)
            plot([0],[0],"w",ms=0,label=predictor_names[j])
            legend(loc="best",fontsize=11,frameon=false,handlelength=0,handletextpad=0,borderpad=0)
            if mod(ji-1,2)==0; yticks([.5,1],fontsize=9); else; yticks([]); end
            if ji in [9,10]; xticks(-0.2:.1:.2,fontsize=9); else; xticks(-0.2:.1:.2,["","","","",""]); end
            #xticks(-0.2:.1:.2,fontsize=8)
            l,u = interval(betas,0.95)
            plot([l,l],ylim(),"r-",markersize=0)
            plot([u,u],ylim(),"r-",markersize=0)
        end
        str = (alpha<Inf? "Coarsened" : "Standard")
        suptitle(latex("\\mathrm{$str posterior c.\\!d.\\!f.\\!s}");fontsize=17,y=0.93)
        #draw()
        savefig("cpp-beta-alpha=$alpha.png",dpi=150)
    end
    save("cpp-results-"*(exclude_mwgt?"wo":"w")*"-mwgt.jld","results",(alphas,k_posteriors,P_posteriors,beta_means,El,Ek))
end

# ____________________________________________________________________________________________________________________________
# Display results

tag = (exclude_mwgt? "" : " (+mother's weight)")

# Plot posteriors on k
figure(5,figsize=(5.5,3.2)); clf()
subplots_adjust(bottom=0.2)
#plot(0:p,VarSel.prior_k(0:p,p),"ko-",label="prior",markersize=8,linewidth=2)
for (i_a,alpha) in enumerate(alphas_for_kplot)
    ind = findfirst(alphas.==alpha)
    lab = "\$\\alpha = " * (alpha<Inf ? repr(round(Int,alpha)) : "\\infty") * "\$"
    plot(0:p,vec(k_posteriors[:,ind]),"$(T10colors[i_a])$(shapes[i_a])-",label=lab,mec="k",ms=8,lw=2)
end
xticks(0:p)
xlim(0,12)
ylim(0,1)
title(latex("\\mathrm{Posterior on k$tag}"),fontsize=17)
xlabel(latex("k  \\mathrm{(\\# of nonzero coefficients)}"),fontsize=17)
ylabel(latex("\\pi_\\alpha(k|\\mathrm{data})"),fontsize=15)
legend(numpoints=1,fontsize=14,labelspacing=0.2,borderaxespad=0.1)
draw()
savefig("cpp-k_posteriors.png",dpi=150)


# Plot calibration curve
figure(6,figsize=(4.5,6.4)); clf()
subplots_adjust(bottom=0.2,left=0.17)
inds = findin(alphas,alphas_for_curve)
plot(Ek[inds],El[inds],"ko-",ms=2,lw=1)
yl = ylim()
X,Y,alpha = Ek[inds][20],El[inds][20],alphas_for_curve[20]
plot(X,Y,"rx",ms=10,mew=2)
text(X,Y+(yl[1]-yl[2])*0.08,latex("\\alpha=$(round(Int,alpha))"),fontsize=14)
X,Y,alpha = Ek[inds][25],El[inds][25],alphas_for_curve[25]
plot(X,Y,"rx",ms=10,mew=2)
text(X,Y+(yl[1]-yl[2])*0.08,latex("\\alpha=$(round(Int,alpha))"),fontsize=14)
plot(Ek[inds],El[inds],"ko-",ms=2,lw=1)
gca()[:ticklabel_format](style="sci",axis="y",scilimits=(0,0),fontsize=8)
yticks(fontsize=9)
xticks(0:7)
title(latex("\\mathrm{Calibration of} \\alpha"*(exclude_mwgt?"":"\\mathrm{$tag}")),fontsize=17)
ylabel(latex("\\hat\\mathrm{E}_\\alpha\\mathrm{(loglik | data)}"),fontsize=15,labelpad=0)
xlabel(latex("\\hat\\mathrm{E}_\\alpha(k | \\mathrm{data})"),fontsize=16,labelpad=0)
draw()
savefig("cpp-alpha-calibration.png",dpi=150)


# Sort coefs by overall probability of inclusion
if true
    order = sortperm(vec(mean(P_posteriors,2)),rev=true)
    nlabel = 11
else
    # Use order from file
    P_post_order = load("cpp-results-wo-mwgt.jld","results")[3]
    order = sortperm(vec(mean(P_post_order,2)),rev=true)
    i_mwgt = findfirst("V_MWGTPP".==predictor_names)
    order[order.>=i_mwgt] += 1
    order = [i_mwgt; order]
    colors = [colors[12]; colors[(1:end).!=12]]
    nlabel = 12
end
names_order = predictor_names[order]
for i in 1:16
    println(i,": ",names_order[i])
end


# Plot posterior probability of inclusion
figure(7,figsize=(8,3.2)); clf()
subplots_adjust(bottom=0.2)
for (i_a,alpha) in enumerate(alphas_for_kplot)
    ind = findfirst(alphas.==alpha)
    lab = "\$\\alpha = " * (alpha<Inf ? repr(round(Int,alpha)) : "\\infty") * "\$"
    plot(1:p,vec(P_posteriors[order,ind]),"$(T10colors[i_a])$(shapes[i_a])-",label=lab,markersize=8,linewidth=2)
end
d_show = 12
xticks(1:d_show)
xlim(0,d_show)
yticks(linspace(0,1,6))
ylim(0,1.1)
title(latex("\\mathrm{Posterior probability of inclusion$tag}"),fontsize=17)
xlabel(latex("j  \\mathrm{(coefficient index)}"),fontsize=17)
ylabel("\$\\Pi_\\alpha(\\beta_j \\neq 0\\,|\\,\\mathrm{data})\$",fontsize=16)
legend(numpoints=1,fontsize=14)
draw()
savefig("cpp-P_posteriors.png",dpi=150)


# Plot posterior means
figure(8,figsize=(8,3.2)); clf()
subplots_adjust(bottom=0.2)
for (i_a,alpha) in enumerate(alphas_for_kplot)
    ind = findfirst(alphas.==alpha)
    lab = "\$\\alpha = " * (alpha<Inf ? repr(round(Int,alpha)) : "\\infty") * "\$"
    plot(1:p,vec(beta_means[order,ind]),"$(T10colors[i_a])$(shapes[i_a])-",label=lab,markersize=8,linewidth=2)
end
d_show = 16
xticks(1:d_show)
xlim(0,d_show)
xlabel("\$j\\,\\,\\mathrm{(coefficient\\,index)}\$",fontsize=17)
ylabel(latex("\\mathrm{E}_\\alpha(\\beta_j | \\mathrm{data})"),fontsize=16)
legend(numpoints=1)
draw()
savefig("cpp-beta_means.png",dpi=150)


# Plot "Coarsening path" (analogous to LASSO path) of posterior means
figure(9,figsize=(6.5,3.2)); clf()
subplots_adjust(bottom=0.2,right=0.75)
#xmin,xmax = log10.(alphas)[[1,end-1]]
xmin,xmax = 2,5
# nshow = 9
for (i_j,j) in enumerate(order)
    if all(beta_means[j,:] .== 0); continue; end
    lab = (i_j<=nlabel? predictor_names[j] : nothing)
    inds = findin(alphas,alphas_for_curve)
    plot(log10.(alphas[inds]),beta_means[j,inds],"-",color=colors[mod(i_j-1,22)+1],linewidth=2,label=lab)
end
plot([xmin,xmax],[0,0],"k-",lw=2)
xlim(xmin,xmax)
title(latex("\\mathrm{Coarsening path$tag}"),fontsize=17)
xlabel(latex("\\log_{10} \\alpha"),fontsize=17)
ylabel(latex("\\mathrm{E}_\\alpha(\\beta_j | \\mathrm{data})"),fontsize=16,labelpad=0)
legend(numpoints=1,bbox_to_anchor=(1.04, 1.04), loc=2, borderaxespad=0., fontsize=10)
draw()
savefig("cpp-beta-path.png",dpi=150)


# ____________________________________________________________________________________________________________________________
# Run LASSO

using Lasso
path = fit(LassoPath,x[:,2:end],y)

# Plot "LASSO path"
xmin,xmax = -1.5,maximum(log10.(path.λ))
figure(10,figsize=(6.5,3.2)); clf()
subplots_adjust(bottom=0.2, right=0.75)
for (i_j,j) in enumerate(order) #[1:nshow])
    b = (predictor_names[j]=="CONST"? path.b0 : collect(path.coefs[j-1,:]))
    lab = (i_j<=nlabel? predictor_names[j] : nothing)
    #if b[findfirst(log10.(path.λ).<xmin)] == 0; continue; end
    plot(log10.(path.λ),b,"-",color=colors[mod(i_j-1,22)+1],linewidth=2,label=lab)
end
plot([xmin,xmax],[0,0],"k-",lw=2)
xlim(xmin,xmax)
title(latex("\\mathrm{LASSO path$tag}"),fontsize=17)
xlabel(latex("\\log_{10} \\lambda"),fontsize=17)
ylabel("\$\\hat\\beta_j\$",fontsize=16,labelpad=-0.2)
legend(numpoints=1,bbox_to_anchor=(1.04, 1.04), loc=2, borderaxespad=0., fontsize=10)
draw()
savefig("cpp-lasso-path.png",dpi=150)




