# MCMC sample for variable selection using Bayarri's mixture of g-priors (robust prior for variable selection).
# Bayarri, Berger, Forte, and Garcia-Donato, The Annals of Statistics, 2012, "Criterion for Bayesian..."
module Bayarri
using Distributions
using VarSel

a,b = 1.0,1.0  # parameters of the prior on g
# (Bayarri et al, Sec. 3.4, recommend a=0.5 and b=1.0, but that makes the prior on g very dispersed, causing difficulty with mixing of g.)

# log of the prior density on g, given z and n (here, k=sum(z)).
logp_g(g,z,n) = (k=sum(z); (g > (b+n)/(1+k)-b? log(a) + a*log((b+n)/(1+k)) - (a+1)*log(g+b) : -Inf))

# log of the prior on z, up to an additive normalization constant.
# logp_z(z) = (k=sum(z); p=length(z); lfact(p-k) + lfact(k) - lfact(p))  # recommendation of Bayarri et al
logp_z(z) = (k=sum(z); p=length(z); lfact(p-k) + lfact(k) - lfact(p) + log(VarSel.prior_k(k,p)))  # modified to match the prior on k

# log(p(y|g,z))
logp_y_given_gz(y,g,z,x) = (@assert(SSR(x,y,z,g)>0); n=length(y); log(0.5) - 0.5*n*log(2*pi) - 0.5*sum(z)*log(1+g) + lgamma(n/2) - 0.5*n*log(0.5*SSR(x,y,z,g)))
SSR(x,y,z,g) = (xz=x[:,z]; xzy=xz'*y; dot(y,y) - (g/(1+g))*dot(xzy, (xz'*xz)\xzy))

# run sampler
function run(x,y,n_total,n_keep)
    # INPUTS
    # x = n-by-p matrix of covariates
    # y = n-by-1 array of responses
    # n_total = number of MCMC samples
    # n_keep = number of MCMC samples to record

    n,p = size(x)
    @assert(length(y)==n)
    sigma_g_prop = 4*n  # standard deviation of the proposal distribution for g

    # initialize
    z = zeros(Bool,p)
    g = n + 1.0
    
    # initialize record-keeping
    @assert(n_total>=n_keep)
    keepers = round.(Int,linspace(1,n_total,n_keep))
    beta_r = zeros(p,n_keep)
    lambda_r = zeros(n_keep)
    g_r = zeros(n_keep)
    i_keep = 1

    for rep = 1:n_total
        # update z (with beta and sigma2 dropped)
        logp_old = logp_y_given_gz(y,g,z,x) + logp_g(g,z,n) + logp_z(z)
        for j = 1:p
            z[j] = 1-z[j]
            logp_new = logp_y_given_gz(y,g,z,x) + logp_g(g,z,n) + logp_z(z)
            p_new = 1/(1+exp(logp_old - logp_new))
            if rand() < p_new
                logp_old = logp_new
            else
                z[j] = 1-z[j]
            end
        end
        
        # update g (with beta and sigma2 dropped)
        g_prop = g + sigma_g_prop*randn()
        if logp_g(g_prop,z,n) != -Inf
            logp_new = logp_y_given_gz(y,g_prop,z,x) + logp_g(g_prop,z,n) + logp_z(z)
            p_accept = min(1,exp(logp_new - logp_old))
            if rand() < p_accept
                g = g_prop
                logp_old = logp_new
            end
        end
        
        # sample sigma2
        A = 0.5*n
        B = 0.5*SSR(x,y,z,g)
        sigma2 = rand(InverseGamma(A,B))
        
        # sample beta
        beta = zeros(p)
        xz = x[:,z]
        xzy = xz'*y
        C = sigma2*(g/(1+g))*inv(xz'*xz)
        C = (C+C')/2  # this is needed since MvNormal demands exact symmetry, and roundoff errors in inv(.) can cause inexactness
        m = C*xzy/sigma2
        beta[z] = rand(MvNormal(m,C))

        # record
        if rep==keepers[i_keep]
            g_r[i_keep] = g
            lambda_r[i_keep] = 1/sigma2
            beta_r[:,i_keep] = beta
            i_keep += 1
        end
    end

    return beta_r,lambda_r,g_r,keepers
end


end # module













