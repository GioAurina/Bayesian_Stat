using LinearAlgebra

include("utilities.jl")
include("priors.jl")
include("proposal_functions.jl")
include("sampling_functions.jl")
include("target_functions.jl")

#' Fit BSA Model
#'
#' @param g data
#' @param n_iter number of posterior samples
#' @param theta0 parameter initializations
#' @param hyperparam hyperparameters

# Function for fitting RPAGP (Random Process Approximate GP)
function fit_model(sites, g, n_iter, theta0, hyperparam)
    chain = Vector{Any}(undef, n_iter)
    chain_f = Vector{Any}(undef, n_iter)
    chain_g = Vector{Any}(undef, n_iter)
    chain_z = Vector{Any}(undef, n_iter)
    chain_beta = Vector{Any}(undef, n_iter)
    chain_gamma = Vector{Any}(undef, n_iter)
    chain_tau = Vector{Any}(undef, n_iter)
    chain_rho = Vector{Any}(undef, n_iter)
    chain_phi = Vector{Any}(undef, n_iter)
    n_time = size(g, 2) #365
    n = size(g, 1) #32

    t = range(1, stop=n_time, length=n_time)
    dist = euclid_dist(sites[:, 1], sites[:, 2], n)
    X = sites[:, 3:6]
    
    # First iteration
    chain[1] = copy(theta0)

    chain_beta[1] = theta0[:beta]
    chain_tau[1] = theta0[:tau]
    chain_gamma[1] = theta0[:gamma]
    chain_rho[1] = theta0[:rho]
    chain_phi[1] = theta0[:phi]
    chain_f[1] = sample_f(g, chain[1], 1)
    Sigma_f = sq_exp_kernel(t, chain[1][:rho], nugget = 1e-9)
    Sigma_f_inv = inv(Sigma_f)
    println(isposdef(Sigma_f))

    chain_g[1] = get_mu_g_matrix(g, chain_f[1], t, chain[1], Sigma_f_inv)

    chain_z[1] = g - chain_g[1]
    
    start = time()

    Sigma_gamma = get_Sigma_gamma(dist, chain[1][:phi])

    # Iterations
    for iter in 2:n_iter
        if (iter/n_iter*100) % 10 == 0.0
            println(" ...", floor(Int, (iter / n_iter) * 100), "%")
        end
        #println(iter)

        curr = copy(chain[iter - 1])
        
        f = sample_f(g, curr, 1)
        
        curr[:beta] = sample_beta(curr, hyperparam, Sigma_gamma, X)
        #println("beta: ", curr[:beta])

        curr[:phi], Sigma_gamma = sample_phi(dist, X, curr, hyperparam)
        #println("phi: ", curr[:phi])
        #isposdef(Sigma_gamma)

        curr[:gamma] = sample_gamma(t, g, f, curr,  Sigma_f, Sigma_f_inv, Sigma_gamma, X, hyperparam)
        #println("gamma: ", curr[:gamma])

        curr[:tau] = sample_tau(t, g, f, curr, hyperparam, Sigma_f, Sigma_f_inv)
        #println("tau: ", curr[:tau])

        curr[:rho] = sample_rho(t, g, f, curr, hyperparam)
        #println("rho: ", curr[:rho])

        Sigma_f = sq_exp_kernel(t, curr[:rho], nugget = 1e-9)
        Sigma_f_inv = inv(Sigma_f)
        #println(isposdef(Sigma_f))
        
        g_hat = get_mu_g_matrix(g, f, t, curr, Sigma_f_inv)
        
        # Residuals computation
        z = g - g_hat
        
        # Saving all the draws of the current iteration
        chain_f[iter] = copy(f)
        chain[iter] = copy(curr)
        chain_g[iter] = copy(g_hat)
        chain_z[iter] = copy(z)
        chain_beta[iter] = copy(curr[:beta])
        chain_gamma[iter] = copy(curr[:gamma])
        chain_tau[iter] = copy(curr[:tau])
        chain_rho[iter] = copy(curr[:rho])
        chain_phi[iter] = copy(curr[:phi])
        
        #println(iter, "  ", chain[iter])
        #println(iter, "   ", curr[:rho])
    end

    println("\n")
    fine = time()
    runtime = fine - start
    println("Tempo di esecuzione: ", runtime)

    return Dict(
        :chain => chain,
        :chain_f => chain_f,
        :chain_g => chain_g,
        :chain_z => chain_z,
        :chain_beta => chain_beta,
        :chain_gamma => chain_gamma,
        :chain_tau => chain_tau,
        :chain_phi => chain_phi,
        :chain_rho => chain_rho,
        :runtime => runtime
    )
end