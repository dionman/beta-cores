s# Functions for AR(k) model

# Generate data from autoregression model with specified noise
# n = sample size (number of time points)
# a = autoregression coefficients
# f_noise(t) = noise at time t
function generate(n,a,f_noise)
    k = length(a)
    x = zeros(n+k)
    for t = k+1:n+k
        x[t] = dot(x[t-1:-1:t-k],a) + f_noise(t)
    end
    x = x[k+1:n+k]
    return x
end

# Log of robustified marginal likelihood for AR(k) model
# x = data
# s = stddev of data, about the mean function
# k = order of model (# of coefs)
# sa = stddev of Normal priors on coefs a_i
# zeta = raise lik to this power
function log_marginal(x,k,s,sa,zeta)
    n = length(x)

    M = zeros(k,k)
    for i = 1:k, j = 1:k
        r = 0.0
        for t = (max(i,j)+1):n
            r += x[t-i]*x[t-j]
        end
        M[i,j] = r/s^2
    end
    v = zeros(k)
    for i = 1:k
        r = 0.0
        for t = (i+1):n
            r += x[t]*x[t-i]
        end
        v[i] = r/s^2
    end

    Lambda = zeta*M + eye(k)/sa^2
    log_m = 0.5*zeta^2*dot(v,Lambda\v) - k*log(sa) - 0.5*logdet(Lambda) - zeta*0.5*n*log(2*pi*s^2) - zeta*0.5*dot(x,x)/s^2

    return log_m
end
