# Analysis of flow cytometry data using mixture model with coarsening.
# __________________________________________________________________________________________
# Settings

# Data settings
datasets_for_testing = 7:12  # datasets to use for testing
varsubset = 3:6  # subset of the variables to use

# Model settings
K = 20  # number of components
gamma0 = 1/(2*K)  # Dirichlet concentration parameter

# Inference settings
alpha = 200  # Choose alpha=200 based on training datasets
n_total = 4000  # total number of MCMC iterations per run
n_burn = 2000  # number of MCMC iterations to discard as burn-in
n_init = round(Int,n_burn/5)  # number of MCMC iterations for initialization with periodic random splits
cutoff = 0.02  # nonnegligible cluster size: 100*cutoff %
from_file = false  # load previous results from file

# __________________________________________________________________________________________
# Supporting functions

using Distributions
using PyPlot
using HDF5, JLD

# Code for mixture model MCMC algorithm
include("core.jl")
# Code for helper functions
include("helper.jl")

# __________________________________________________________________________________________
# Data

# Load the specified data set from the FlowCAP-I GvHD collection
function load_GvHD(d::Int)
    labels_file = "FlowCAP-I/GvHD/labels$(d).csv"
    data_file = "FlowCAP-I/GvHD/sample$(d).csv"
    labels = vec(readdlm(labels_file,',',Int))
    data,varnames = readdlm(data_file,',',Float64; header=true)
    return data,labels,varnames
end

# __________________________________________________________________________________________
# Visualize each data set using pairwise scatterplots

if false
    for d in 1:12
        data,labels,varnames = load_GvHD(d)
        figure(1,figsize=(12,12)); clf()
        title("Data set #$d")
        pairwise_scatterplots(data[:,varsubset],labels+1,varnames[varsubset],T10colors;ms=0.25)
        drawnow()
        savefig("d-$d-scatter.png",dpi=200)
    end
end

if true
    for d in [7,10]
        data,labels,varnames = load_GvHD(d)
        x = data[:,varsubset]
        labels += 1
        varnames = varnames[varsubset]
        L = unique(labels)
        for (i,j) in [(1,2),(2,3),(3,4)]
            figure(1,figsize=(5.5,5.5)); clf()
            for l in L
                plot(x[labels.==l,j],x[labels.==l,i],"o",color=T10colors[l],ms=0.6)
            end
            title("$(varnames[i]) \$\\mathrm{vs}\$ $(varnames[j])  \$\\mathrm{(Manual)}\$",fontsize=17)
            xticks(fontsize=10)
            yticks(fontsize=10)
            savefig("mixcyto-clusters-d=$d-$(i)vs$(j)-manual.png",dpi=150)
        end
    end
end


# __________________________________________________________________________________________
# Calibration runs to choose alpha based on training datasets

if true
    # Calibration settings
    datasets_for_calibration = 1:6
    alphas = [20; 50; 100; 150; 200; 250; collect(300:100:600)]
    nreps = 1

    n_alphas = length(alphas)
    n_datasets = length(datasets_for_calibration)
    Fs = zeros(n_alphas,n_datasets,nreps)
    for (i_d,d) in enumerate(datasets_for_calibration)
        # Load data
        println("d = $d")
        data,labels,varnames = load_GvHD(d)
        x = data[:,varsubset]'
        n = size(x,2)
        # Data-dependent hyperparameters
        mu0,L0,nu0,V0 = hyperparameters(x)
        
        for (i_a,a) in enumerate(alphas)
            println("alpha = $a")
            zeta = a/(a + n)
            
            for rep = 1:nreps
                # Run MCMC sampler
                tic()
                N_r,p_r,mu_r,L_r,z_r,zmax_r = sampler(x,n_total,n_init,K,gamma0,mu0,L0,nu0,V0,zeta)
                toc()
                use = n_burn+1:n_total  # subset of samples to use
                
                # Compute F-measures
                L = find(labels.!=0)
                Fm = [F_measure(labels[L],zmax_r[i][L])[1] for i=1:n_total]
                Fs[i_a,i_d,rep] = mean(Fm[use])

                # Display traceplot of F-measures
                figure(50); clf()
                title("Traceplot of F measure (d=$d, a=$a, rep=$rep)")
                plot(1:length(Fm),Fm,"k.-",ms=0.1,lw=0.5)
                # savefig("F-traceplot-d=$d-a=$a-rep=$rep.png", dpi=150)
                # drawnow()
            end
        end
    end

    # Plot Fs
    figure(2,figsize=(8,3.2)); clf()
    for (i_d,d) in enumerate(datasets_for_calibration)
        plot(alphas,mean(Fs,3)[:,i_d,1],label=repr(d),"$(shapes[i_d])-",lw=1,ms=4)
    end
    plot(alphas,mean(Fs,[2,3])[:],"k--",lw=2)
    ylim(ylim()[1],1.0)
    xlim(0,xlim()[2])
    xticks(alphas)
    subplots_adjust(bottom=0.2)
    title(latex("\\mathrm{Calibration of }\\alpha\\mathrm{ using training datasets}"),fontsize=17)
    ylabel(latex("\\mathrm{F measure}"),fontsize=16)
    xlabel(latex("\\alpha"),fontsize=16)
    legend(ncol=3)
    draw()
    savefig("mixcyto-calibration.png",dpi=200)

    # Save Fs to file
    save("calibration.jld","Fs",Fs)
end



# __________________________________________________________________________________________
# Run algorithm on test datasets

n_datasets = length(datasets_for_testing)
Fs = zeros(n_datasets,2)
kp_posteriors = zeros(K+1,n_datasets,2)
algorithm_labels = Array{Array{Int,1},2}(n_datasets,2)
for (i_d,d) in enumerate(datasets_for_testing)
    # Load data
    data,labels,varnames = load_GvHD(d)
    x = data[:,varsubset]'
    n = size(x,2)

    # Data-dependent hyperparameters
    mu0,L0,nu0,V0 = hyperparameters(x)

    figure(10+i_d,figsize=(8,3.2)); clf()
    subplots_adjust(bottom=0.2)
    figure(20+i_d,figsize=(8,3.2)); clf()
    subplots_adjust(bottom=0.2)

    # Run for coarsened and standard posteriors
    for (i_a,a) in enumerate([Inf,alpha])
        srand(0) # reset RNG
        # Run MCMC sampler
        tic()
        zeta = (a<Inf? a/(a+n) : 1.0)
        N_r,p_r,mu_r,L_r,z_r,zmax_r = sampler(x,n_total,n_init,K,gamma0,mu0,L0,nu0,V0,zeta)
        toc()
        use = n_burn+1:n_total  # subset of samples to use
        n_use = length(use)
        algorithm_labels[i_d,i_a] = zmax_r[end]

        # Compute the posterior on the number of nonnegligible clusters
        kp_r = map(N->sum(N.>n*cutoff), N_r)
        counts,~ = histogram(kp_r[use],-0.5:1:K+0.5)
        kp_posterior = counts/n_use
        kp_posteriors[:,i_d,i_a] = kp_posterior
        
        # Compute the F-measure between the manual clustering and algorithmic clustering
        L = find(labels.!=0)
        F = [F_measure(labels[L],z_r[i][L])[1] for i=1:n_total]
        println("F-measure =           ",mean(F[use]))
        Fm = [F_measure(labels[L],zmax_r[i][L])[1] for i=1:n_total]
        println("F-measure (zmax) =    ",mean(Fm[use]))
        Fs[i_d,i_a] = mean(Fm[use])

        # Find permutation of labels to try to match
        lm = labels  # manual labels
        la = zmax_r[end]  # algorithm labels
        la[L] = match_labels(lm[L],la[L])
        la[labels.==0] = 0

        # Display clusters at the last step of the sampler
        figure(3,figsize=(12,12)); clf()
        pairwise_scatterplots(data[:,varsubset],la+1,varnames[varsubset],[T10colors;colors[end-1:-1:1]];ms=0.25)
        #savefig("mixcyto-clusters-model-d=$d-a=$a.png",dpi=200)
        # drawnow()
        L = unique(la)
        C = [T10colors;colors[end-1:-1:1]]
        for (i,j) in [(1,2),(2,3),(3,4)]
            figure(30+i_a,figsize=(5.5,5.5)); clf()
            for l in L
                plot(data[la.==l,varsubset[j]],data[la.==l,varsubset[i]],"o",color=C[l+1],ms=0.6)
            end
            label = (a<Inf? "Coarsened,  \\alpha=$(round(Int,a))" : "Standard")
            title("$(varnames[varsubset][i]) \$\\mathrm{vs}\$ $(varnames[varsubset][j])  \$\\mathrm{($label)}\$",fontsize=17)
            xticks(fontsize=10)
            yticks(fontsize=10)
            savefig("mixcyto-clusters-d=$d-$(i)vs$(j)-a=$(a<Inf? round(Int,a) : a).png",dpi=150)
        end

        # Display traceplot of F-measures
        figure(4,figsize=(8,3.2)); clf()
        title("Traceplot of F measure (zmax)")
        plot(1:length(Fm),Fm,"k.-",ms=0.1,lw=0.5)
        #savefig("F-traceplot-d=$d-a=$a.png", dpi=150)
        # drawnow()
    end
    close("all")
    save("results.jld","Fs",Fs,"kp_posteriors",kp_posteriors,"algorithm_labels",algorithm_labels)
    writedlm("results-F.tsv",Fs)
    println("F measures:"); display(Fs)
end



nothing

