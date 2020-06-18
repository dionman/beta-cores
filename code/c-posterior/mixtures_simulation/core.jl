# MCMC sampler for mixture model inference using coarsening, using univariate normal components

normpdf(x,m,s) = (r=(x-m)./s; exp.(-0.5*r.*r)./(sqrt(2*pi)*s))
count!(z,N) = (fill!(N,0); for zi in z; N[zi]+=1; end; N)

function sampler(x,n_samples,n_init,K,gamma0,mu0,sigma0,a0,b0,zeta; mode="standard")
    n = length(x)
    subset = (mode=="subset"? randperm(n)[1:round(Int,n*zeta)] : (1:n))
    power = (mode=="downweight"? zeta : 1.0)

    # Initialize
    mu = rand(Normal(mu0,sigma0),K)
    sigma = sqrt.(rand(InverseGamma(a0,b0),K))
    p = rand(Dirichlet(K,gamma0))
    z = zeros(Int,n)

    # Record-keeping
    N_r = zeros(Int,K,n_samples)
    p_r = zeros(Float64,K,n_samples)
    mu_r = zeros(Float64,K,n_samples)
    sigma_r = zeros(Float64,K,n_samples)
    logw = zeros(Float64,n_samples)
    L_r = zeros(Float64,n_samples)

    # Temporary variables
    N = zeros(Int,K)
    sumx = zeros(Float64,K)
    sumxx = zeros(Float64,K)
    lik = zeros(Float64,n)

    # MCMC
    for iter = 1:n_samples
        # update z
        for i = 1:n
            r = p.*normpdf(x[i],mu,sigma)
            z[i] = rand(Categorical(r/sum(r)))
            lik[i] = sum(r)
        end

        # randomly split clusters during initialization period
        if ((iter%10)==0) && (iter < n_init)
            fill!(N,0); for i=1:n; k=z[i]; N[k]+=1; end
            unused = find(N.==0)
            for k in sortperm(N; rev=true)
                if (N[k]==0) || isempty(unused); break; end
                z[z.==k] = rand([k,pop!(unused)],N[k])
            end
        end
        
        # compute statistics N,sumx,sumxx
        fill!(N,0)
        fill!(sumx,0.0)
        fill!(sumxx,0.0)
        for i in subset 
            k = z[i]
            N[k] += 1
            sumx[k] += x[i]
            sumxx[k] += x[i]*x[i]
        end

        # update p
        p = rand(Dirichlet(gamma0 + power*N))

        # update mu
        for k = 1:K
            lambda = 1/sigma0^2 + power*N[k]/sigma[k]^2
            m = (mu0/sigma0^2 + (1/sigma[k]^2)*power*sumx[k]) / lambda
            s = 1/sqrt(lambda)
            mu[k] = rand(Normal(m,s))
        end
        # update sigma
        for k = 1:K
            ap = a0 + 0.5*power*N[k]
            bp = b0 + 0.5*power*(sumxx[k] - 2*sumx[k]*mu[k] + N[k]*mu[k]*mu[k])
            sigma[k] = sqrt(rand(InverseGamma(ap,bp)))
        end

        # compute unnormalized log-weight for importance sampling
        logp_target = zeta*sum(log.(lik))
        L_r[iter] = sum(log.(lik))
        if mode=="subset"
            logp_sample = sum(log.(lik[subset]))
            logw[iter] = logp_target - logp_sample
        end

        # record
        N_r[:,iter] = N
        p_r[:,iter] = p
        mu_r[:,iter] = mu
        sigma_r[:,iter] = sigma
    end
    return N_r,p_r,mu_r,sigma_r,logw,L_r
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

