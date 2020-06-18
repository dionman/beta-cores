# MCMC sampler for mixture model inference using coarsening, using multivariate normal components

# Helper functions
logsumexp(x) = (m = maximum(x); m == -Inf ? -Inf : log.(sum(exp.(x-m))) + m)
count!(z,N) = (fill!(N,0); for zi in z; N[zi]+=1; end; N)
invSPD(A) = inv(cholfact(Hermitian(A)))

# Data-dependent hyperparameters
function hyperparameters(x)
    d,n = size(x)
    mu0 = vec(mean(x,2))  # prior mean of the component means
    C_hat = (x.-mu0)*(x.-mu0)'/n  # empirical covariance matrix of the data
    L0 = invSPD(C_hat)  # prior precision matrix of the component means
    nu0 = d  # prior degrees of freedom of the component precision matrices
    V0 = invSPD(C_hat)/nu0  # prior scale matrix of the component precision matrices
    return mu0,L0,nu0,V0
end

# Prior on component parameters:
#   mu_k ~ N(mu0,inv(L0))
#   L_k ~ Wishart(nu0,V0)

function sampler(x,n_samples,n_init,K,gamma0,mu0,L0,nu0,V0,zeta)
    d,n = size(x)

    # Initialize
    N = zeros(Int,K)
    p = rand(Dirichlet(K,gamma0))
    mu = [rand(MvNormal(mu0,invSPD(L0))) for k=1:K]
    L = [rand(Wishart(nu0,(V0+V0')/2)) for k=1:K]
    z = rand(Categorical(p),n)
    zmax = copy(z)
    
    # Precompute some stuff
    L0mu0 = L0*mu0
    inv_V0 = invSPD(V0)
    ll = zeros(K,n)
    loglik = zeros(n)

    # Record-keeping
    N_r = Array{Int,1}[]
    p_r = Array{Float64,1}[]
    mu_r = Array{Array{Float64,1},1}[]
    L_r = Array{Array{Float64,2},1}[]
    z_r = Array{Int,1}[]
    zmax_r = Array{Int,1}[]

    # MCMC
    for iter = 1:n_samples
        # update z
        ll .= log.(p)
        for k = 1:K
            y = x .- mu[k]
            ll[k,:] += 0.5*logabsdet(L[k])[1] - 0.5*d*log(2*pi) - 0.5*vec(sum(y.*(L[k]*y),1))
        end
        for i = 1:n
            lse = logsumexp(ll[:,i])
            r = exp.(ll[:,i] - lse)
            z[i] = rand(Categorical(r/sum(r)))
            zmax[i] = indmax(r)
            loglik[i] = lse
        end
        count!(z,N)

        # randomly split clusters during initialization period
        if ((iter%10)==0) && (iter < n_init)
            unused = find(N.==0)
            for k in sortperm(N; rev=true)
                if (N[k]==0) || isempty(unused); break; end
                z[z.==k] = rand([k,pop!(unused)],N[k])
            end
            count!(z,N)
        end

        # update p
        p = rand(Dirichlet(gamma0 + zeta*N))

        # update mu
        for k = 1:K
            L_c = L0 + zeta*N[k]*L[k]
            inv_L_c = invSPD(L_c)
            mu_c = inv_L_c*(L0mu0 + zeta*L[k]*vec(sum(x[:,z.==k],2)))
            mu[k] = rand(MvNormal(mu_c,inv_L_c))
        end
        # update L
        for k = 1:K
            nu_c = nu0 + zeta*N[k]
            y = x[:,z.==k] .- mu[k]
            V_c = invSPD(inv_V0 + zeta*y*y')
            L[k] = rand(Wishart(nu_c,V_c))
        end
        
        if (iter%100)==0; println(iter); end
        
        # record
        push!(N_r, copy(N))
        push!(p_r, copy(p))
        push!(mu_r, deepcopy(mu))
        push!(L_r, deepcopy(L))
        push!(z_r, copy(z))
        push!(zmax_r, copy(zmax))
    end
    return N_r,p_r,mu_r,L_r,z_r,zmax_r
end


# Compute histogram with the specified bin edges,
# where x[i] is in bin j if edges[j] < x[i] <= edges[j+1].
function histogram(x, edges=[]; n_bins=50, weights=ones(length(x)))
    if isempty(edges)
        mn,mx = minimum(x),maximum(x)
        r = mx-mn
        edges = linspace(mn-r/n_bins, mx+r/n_bins, n_bins)
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

