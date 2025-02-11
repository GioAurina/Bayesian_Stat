using Distributions, LinearAlgebra

prior = Dict()


#' Prior for tau (latency).
#'
#' @param tau tau value.
#' @param tau_prior_sd SD of prior distribution.
prior[:tau] = function prior_tau(tau, tau_prior_sd)
    result = 0.0
    n = length(tau)
    Sigma = tau_prior_sd^2 * (I(n) - ones(n, n) * (1 / (n + 1)))
    mvn = MvNormal(zeros(n), Sigma)
    result += logpdf(mvn, tau[1:n])
    return result
end



prior[:tau_i] = function prior_tau_i(tau_i, tau_prior_sd)
    uvn = Normal(0, tau_prior_sd)
    result = logpdf(uvn, tau_i)
    return result
end

#=
prior[:tau] = function prior_tau(tau, tau_prior_sd)
    result = 0.0
    n = length(tau)
    for i in 1:n
        result += logpdf(Normal(0, tau_prior_sd), tau[i])
    end
    return result
end
=#


#' Prior for rho (temporal GP length scale).
#'
#' @param rho rho value.
#' @param rho_prior_shape Shape parameter of prior.
#' @param rho_prior_scale Scale parameter of prior.
prior[:rho] = function prior_rho(rho, rho_prior_shape, rho_prior_scale)
    gamma_dist = Gamma(rho_prior_shape, rho_prior_scale)
    result = logpdf(gamma_dist, rho)
    return result
end


#' Prior for phi (spatial GP length scale).
#'
#' @param phi phi value.
#' @param phi_prior_shape Shape parameter of prior.
#' @param phi_prior_scale Scale parameter of prior.
prior[:phi] = function prior_phi(phi, phi_prior_shape, phi_prior_scale)
    gamma_dist = Gamma(phi_prior_shape, phi_prior_scale)
    result = logpdf(gamma_dist, phi)
    return result
end


#' Prior for gamma.
#'
#' @param gamma gamma value.
#' @param phi_prior_shape Shape parameter of prior.
#' @param phi_prior_scale Scale parameter of prior.
prior[:gamma] = function prior_gamma(gamma, Sigma_gamma, mu_gamma)
    n = length(gamma)
    mvn = MvNormal(mu_gamma, Sigma_gamma)
    result = logpdf(mvn, gamma[1:n])
    return result
end


#' Marginal prior for beta_i.
#' @param beta_i beta_i value.
prior[:beta_i] = function prior_beta_i(beta_i)
    mvn = Normal(0, 1)
    result = logpdf(mvn, beta_i)
    return result
end