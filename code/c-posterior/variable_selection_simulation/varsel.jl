# MCMC sampler for variable selection using the power posterior.
module VarSel
using Distributions

# w ~ Beta(aw,bw)
a_w(p) = 1.0
b_w(p) = 2*p
# prior on beta's given w: N(0,1/L0) w.p. w, and 0 w.p. 1-w.
sigma = 1.0
L0 = 1/sigma^2

# prior on lambda: Gamma(a,b)
a,b = 1.0,1.0

# induced prior on # of nonzero coefs k: BetaBinomial(p,aw,bw)
log_betabern(k,n,a,b) = lgamma(a+k) + lgamma(b+n-k) - lgamma(a+b+n) - lgamma(a) - lgamma(b) + lgamma(a+b)
log_betabin(k,n,a,b) = lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1) + log_betabern(k,n,a,b)
prior_k(k,p) = exp(log_betabin(k,p,a_w(p),b_w(p)))

# run sampler
function run(x,y,n_total,n_keep,zeta)
    # INPUTS
    # x = n-by-p matrix of covariates
    # y = n-by-1 array of responses
    # n_total = number of MCMC samples
    # n_keep = number of MCMC samples to record
    # zeta = power to raise likelihood to

    n,p = size(x)
    @assert(length(y)==n)
    aw = a_w(p)
    bw = b_w(p)

    # initialize
    if true
        beta = zeros(p)
    else # sample from prior
        w = rand(Beta(aw,bw))
        z = (rand(p).<w)
        beta = z.*randn(p)/sqrt(L0)
    end
    lambda = 1.0
    delta = y - x*beta
    k = sum(beta.!=0)
    sx = vec(sum(x.^2,1))
    
    # initialize record-keeping
    @assert(n_total>=n_keep)
    keepers = round.(Int,linspace(1,n_total,n_keep))
    beta_r = zeros(p,n_keep)
    lambda_r = zeros(n_keep)
    i_keep = 1

    for rep = 1:n_total
        # update lambda
        A = a + 0.5*n*zeta
        B = b + 0.5*sum(delta.^2)*zeta
        lambda = rand(Gamma(A,1/B))
        
        # update beta
        for j = 1:p
            delta += beta[j]*x[:,j]
            k -= round(Int,beta[j]!=0)
            L = L0 + zeta*lambda*sx[j]
            M = (zeta*lambda/L)*dot(delta,x[:,j])
            log_p0 = log(bw + p-1 - k)  # unnormalized prob of 0
            log_p1 = log(aw + k) + 0.5*log(L0) - 0.5*log(L) + 0.5*L*M*M  # unnormalized prob of nonzero
            p0 = 1/(1 + exp(log_p1 - log_p0))
            if rand() < p0
                beta[j] = 0.0
            else
                beta[j] = randn()/sqrt(L) + M
                delta -= beta[j]*x[:,j]
                k += 1
            end
        end
        
        # record
        if rep==keepers[i_keep]
            lambda_r[i_keep] = lambda
            beta_r[:,i_keep] = beta
            i_keep += 1
        end
    end

    return beta_r,lambda_r,keepers
end





end # module













