# Functions for multivariate skew-normal distributions
module Skew

mvnrnd(n,mu,C) = sqrtm(C)*randn(size(C,1),n) .+ mu

# Generate n random samples from SkewNormal(Omega,alpha).
#    Omega = d-by-d "correlation" matrix (a covariance matrix with ones on the diagonal).
#    alpha = length d vector of skew parameters.
# (This implements Proposition 1 of Azzalini & Capitanio (1999).)
function skewrndNd(n,Omega,alpha)
    delta = (Omega*alpha)/sqrt(1+dot(alpha,Omega*alpha))
    x = mvnrnd(n,0,[1 delta'; delta Omega])
    return sign.(x[1,:]').*x[2:end,:]
end

# Generate n random samples from SkewNormal(xi,C,alpha).
#    xi = length d vector of location parameters (it is not necessarily the mean of the resulting samples).
#    C = d-by-d posdef matrix (it is not necessarily the covariance matrix of the resulting samples).
#    alpha = length d vector of skew parameters.
# (This implements the location-scale extension in Section 5.1 of Azzalini & Capitanio (1999).)
function skewrndNd(n,xi,C,alpha)
    s = sqrt.(diag(C))
    Omega = (1.0./s) .* C .* (1.0./s)'
    return xi .+ s.*skewrndNd(n,Omega,alpha)
end

# Compute the mean of Z ~ SkewNormal(Omega,alpha).
# (See Equation (4) of Azzalini & Capitanio (1999).)
mu(Omega,alpha) = (delta = (Omega*alpha)/sqrt(1+dot(alpha,Omega*alpha)); sqrt(2/pi)*delta)

# Compute the covariance matrix of Z ~ SkewNormal(Omega,alpha).
# (See Equation (4) of Azzalini & Capitanio (1999).)
Cov(Omega,alpha) = (m = mu(Omega,alpha); Omega-m*m')

# Generate n random samples from SkewNormal(Omega,alpha) and transform them so that each entry has zero mean and unit variance.
skewrndNormalized(n,Omega,alpha) = (skewrndNd(n,Omega,alpha) .- mu(Omega,alpha))./sqrt.(diag(Cov(Omega,alpha)))

end

