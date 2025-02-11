using Distributions, LinearAlgebra, Random, Printf, DataFrames, StatsBase, ToeplitzMatrices

#' Generate data.
#'
#' @param n Number of trials.
#' @param n_time Number of time points.
#' @param theta Named list of parameter values.
function generate_data(sites, n, K, n_time, theta)

    X = sites[:, 3:end] # Design matrix: caratteristiche del sito
    coords = sites[:, 1:2] # coordinate geografiche
    
    # Genera i valori di t
    t = range(1, stop=n_time, length=n_time)

    f = zeros(Float64, K, n_time)
    g = zeros(Float64, n, K, n_time)
    gamma = zeros(Float64, n, K)

    dist = euclid_dist(coords[:, 1], coords[:, 2], n)

    for k in 1:K
        theta_k = theta[k]
        Sigma_f = sq_exp_kernel(t, theta_k[:rho]; nugget=1e-6)
        Sigma_f_inv = inv(Sigma_f)
        # Genera il vettore f dalla distribuzione normale multivariata con media zero e matrice varianza Sigma_f
        f[k, :] = rand(MvNormal(zeros(n_time), Sigma_f))
         
        Sigma_gamma = get_Sigma_gamma(dist, theta_k[:phi])
        gamma[:, k] = rand(MvNormal(X*theta_k[:beta], Sigma_gamma))
        theta_k[:gamma] = gamma[:, k]
        # Loop attraverso ciascuna colonna i
        for i in 1:n
            Sigma_i = get_Sigma_i(i, t, theta_k)

            # Calcola mu come Sigma_i * Sigma_f_inv * f
            g[i, k, :] = get_mu_g(i, t, f[k,:], theta_k, Sigma_f_inv)
        end

    end
    
    return Dict(:g => g, :f => f, :gamma => gamma)
end


#' Generate covariance matrix for square exponential kernel.
#'
#' @param t Vector of time points.
#' @param rho Length scale.
#' @param alpha Amplitude.
#' @param nugget Covariance nugget.
function sq_exp_kernel(t, rho; alpha=1, nugget=0.0)
    n_time = length(t)
    K = Matrix{Float64}(undef, n_time, n_time)

    for i in 1:n_time
        for j in 1:n_time
            K[i, j] = alpha^2 * exp(- (rho)^2 / 2 * (t[i] - t[j])^2)
        end
    end
    
    K += nugget .* I(n_time)   
    K = (K + K')/2
    
    return K
end



#' Generate covariance matrix for square exponential kernel.
#'
#' @param D matrix of Euclidean distances.
#' @param phi Length scale.
#' @param alpha Amplitude.
#' @param nugget Covariance nugget.
function get_Sigma_gamma(D, phi; alpha=1, nugget=1e-6)
    n_stations = size(D, 1)

    K = Matrix{Float64}(undef, n_stations, n_stations)

    for i in 1:n_stations
        for j in 1:n_stations
            K[i, j] = alpha^2 * exp(-phi^2 / 2 * (D[i,j])^2)
        end
    end
    
    K += nugget .* I(n_stations)   
    K = (K + K')/2
    
    #println(isposdef(K))

    return K
end




function get_Sigma_i(i, t, theta)
    n_time = length(t)
    K = Matrix{Float64}(undef, n_time, n_time)

    for ii in 1:n_time
        for jj in 1:n_time
            K[ii, jj] = exp(-theta[:rho]^2 / 2 * (t[ii] - t[jj] - theta[:tau][i])^2)
        end
    end

    #return theta[:gamma] .* K
    return K
end






# Calcolo della distanza euclidea
function euclid_dist(x::Vector{Float64}, y::Vector{Float64}, n::Int)
    
    R = 6371.0
    dist = zeros(Float64, n, n)

    for i in 1:n
        for j in 1:n
            lat1, lon1 = deg2rad(x[i]), deg2rad(y[i])
            lat2, lon2 = deg2rad(x[j]), deg2rad(y[j])

            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            a = sin(dlat / 2)^2 + cos(lat1) * cos(lat2) * sin(dlon / 2)^2
            dist[i, j] = 2 * R * asin(min(1, sqrt(a)))  # Usa arcsin (asin in Julia)
            
        end
    end
    return dist
end





#' Get g_ki mean
#'
#' @param i Subject index (scalar, 1 < i < n).
#' @param t Vector of time instances.
#' @param f Vector of values for f.
#' @param theta Named list of parameter values.
#' @param gamma Vector of values for gamma.
#' @param Sigma_f_inv Inverse covariance matrix of f.
function get_mu_g(i, t, f, theta, Sigma_f_inv)
    
    Sigma_i = get_Sigma_i(i, t, theta)
    
    # Calcola mu come il prodotto Sigma_i * Sigma_f_inv * f
    mu = exp(theta[:gamma][i]) * Sigma_i * Sigma_f_inv * f
    return mu[:,1]
end



#' Get mu_g for all trials, output in matrix form
#'
#' @param g Matrix of observed trial data.
#' @param f Vector of f values.
#' @param theta Parameter values.
#' @param Sigma_f_inv Inverse covariance matrix of f (n_time x n_time).
function get_mu_g_matrix(g, f, t, theta, Sigma_f_inv)
    # Inizializza la matrice g_hat con il numero di righe e colonne di g
    g_hat = Matrix{Float64}(undef, size(g, 1), size(g, 2))

    # Popola ogni colonna di g_hat utilizzando la funzione get_mu_g
    for i in 1:size(g, 1)
        g_hat[i, :] = get_mu_g(i, t, f, theta, Sigma_f_inv)
    end
    
    return g_hat
end



#' Get Sigma_g_i (Section 4.6.2)
#' @param beta_i Trial specific amplitude beta_i
#' @param Sigma_f Covariance matrix of f. (n_time x n_time)
#' @param Sigma_nu Covariance AR process (n_time x n_time)
function get_Sigma_g_i(i, t, theta, Sigma_f, Sigma_f_inv)

    Sigma_i = get_Sigma_i(i, t, theta)

    K = (exp(theta[:gamma][i])^2) .* (Sigma_f - Sigma_i * Sigma_f_inv * (Sigma_i)' ) + 0.00005 .* I(size(Sigma_f, 1))
    # Simmetrizzo
    K  = (K  + K') / 2

    #println("staz: ", i, "posdef:", isposdef(K))
    #println("staz: ", i, "symmetric:", issymmetric(K))
    return K
end


#' Get Sigma_y_i f (Section 4.6.2)
#' @param i trial specific amplitude beta_i
#' @param x Kernel GP
#' @param theta Named list of parameter values
#' @param Sigma_f Covariance matrix of f. (n_time x n_time).
#' @param Sigma_f_inv Inverse covariance matrix of f (n_time x n_time).
#' @param Sigma_nu Covariance AR process (n_time x n_time).
# function getSigma_g_i_f(i, x, theta, Sigma_f, Sigma_f_inv)
#     # Calcola Sigma_y_i
#     Sigma_g_i = get_Sigma_g_i(theta[:gamma][i], Sigma_f)
    
#     # Calcola Sigma_i usando get_Sigma_i (assumendo che questa funzione sia già definita)
#     Sigma_i = get_Sigma_i(x, Dict(:rho => theta[:rho], :tau => theta[:tau][i], :gamma => theta[:gamma][i]))

#     # Calcola Sigma_i
#     Sigma_i = Sigma_g_i - (Sigma_i)' * Sigma_f_inv * Sigma_i
    
#     # Rendi Sigma_i simmetrica
#     Sigma_i = (Sigma_i + (Sigma_i)') / 2
    
#     return Sigma_i
# end




# function get_sigma_loggamma_g(n, n_time, f, Sigma_gamma, mu_loggamma, mu_gamma, current, Sigma_f_inv)

#     x = range(1, stop=n_time, length=n_time)

#     f_matrix = zeros(n, n * n_time)

#     mu_f_tau = zeros(n_time) #da modificare se GP troncato

#     mu_f = zeros(n_time) #da modificare se GP troncato

#     for i in 1:n

#         Sigma_f_tau_i_and_f = get_Sigma_i(x, Dict(:rho => current[:rho], :tau => current[:tau][i], :gamma => 1.0))

#         mu_f_tau_i_dato_f = mu_f_tau + Sigma_f_tau_i_and_f * Sigma_f_inv * (f - mu_f)

#         f_matrix[i, ((i-1)*n_time)+1:i*n_time] = mu_f_tau_i_dato_f
#     end

#     loggamma_gamma_matrix = zeros(n, n)

#     for i in 1:n
#         for j in 1:n

#             loggamma_gamma_matrix[i,j] = exp(mu_loggamma[j] + Sigma_gamma[j,j]/2) * (mu_loggamma[i] + Sigma_gamma[i,j])
            
#         end
#     end

#     return loggamma_gamma_matrix * f_matrix - mu_loggamma * mu_gamma' * f_matrix, f_matrix', f_matrix' * mu_gamma


# end




#' Summary output MCMC
#' @param results output from fit_RPAGP
#' @param dat_trials simulated data in long format 
#' @param y data
#' @param burn_in burn in 
function getSummaryOutput(results, dat_trials, g, burn_in)
    n = size(g, 1)  # Numero di trials
    n_time = size(g, 2)  # Numero di time points
    n_iter = length(results[:chain])  # Numero di iterazioni
    n_final = n_iter - burn_in  # Iterazioni rimanenti dopo il burn-in
    
    # Ottieni le stime delle prove singole
    GSTE = getSingleTrialEstimates(results, burn_in, n, n_time)
    
    g_hat = GSTE[:g_hat]

    f_hat = GSTE[:chain_f_burned]

    # Definisci i quantili
    probs = [0.025, 0.5, 0.975]
    
    # Inizializza un array per i quantili
    g_hat_quantiles = Array{Float64}(undef, 3, n, n_time)
    
    # Calcola i quantili per ogni time point e trial
    for ii in 1:n
        for t in 1:n_time
            g_hat_quantiles[:, ii, t] = quantile(g_hat[ii, t, :], probs)
        end
    end
    
    # Estrai i quantili
    lower = Vector(reshape(g_hat_quantiles[1, :, :], n_time * n))
    median = Vector(reshape(g_hat_quantiles[2, :, :], n_time * n))
    upper = Vector(reshape(g_hat_quantiles[3, :, :], n_time * n))
    mean_f= vec(mean(f_hat, dims=2))

    # Aggiungi i quantili ai dati esistenti
    out = DataFrame(dat_trials)
    out.lwr = lower
    out.med = median
    out.upr = upper

    
    return out, mean_f, g_hat_quantiles
end




#' get singleTrialEstimates 
#' @param results output from fit_RPAGP
#' @param burn_in burn_in period
function getSingleTrialEstimates(results, burn_in, n, n_time)
    n_iter = length(results[:chain])  # Numero di iterazioni
    n_final = n_iter - burn_in  # Iterazioni rimanenti dopo il burn-in
    
    # - Estrazione di beta
    chain_gamma_burned = zeros(n, n_final)  # Matrice per beta
    ss = 1
    for tt in (burn_in+1):n_iter
        chain_gamma_burned[:, ss] = results[:chain][tt][:gamma]  # gamma dalla catena
        ss += 1
    end
    
    # - Estrazione di f
    chain_f_burned = zeros(n_time, n_final)  # Matrice per f
    ss = 1
    for tt in (burn_in+1):n_iter
        chain_f_burned[:, ss] = results[:chain_f][tt]  # f dalla catena
        ss += 1
    end
    
    # - Calcolo di y_hat
    g_hat = zeros(n, n_time, n_final)  # Array per le stime
    for tt in 1:n_final
        for ii in 1:n
            g_hat[ii, :, tt] = chain_gamma_burned[ii, tt] * chain_f_burned[:, tt]
        end
    end
    
    return Dict(:g_hat => g_hat, :chain_f_burned => chain_f_burned)
end



#' Transform an upper triangular matrix to symmetric
#' @param m upper triangular matrix
function ultosymmetric(m)
    m = m + transpose(m) - Diagonal(Diagonal(m))
    return m
end

#' Wrap data as we wish, like getSummaryOutput
function wrap_data(result, burn_in, n_iter, twoD_flag)
    
    var_cut = result[burn_in:n_iter]
    probs = [0.025, 0.5, 0.975]

    # Calcola i quantili per ogni time point e trial
    if twoD_flag == true
        
        var_length = length(var_cut[1])
        var_quantiles = Array{Float64}(undef, 3, var_length)
        for t in 1:var_length
            var_quantiles[:, t] = quantile([obs[t] for obs in var_cut], probs)
        end

        # Estrai i quantili
        lower = var_quantiles[1, :]
        median = var_quantiles[2, :]
        upper = var_quantiles[3, :]

    else 

        var_length1 = length(var_cut[1][:,1]) # sites
        var_length2 = length(var_cut[1][1,:]) # time
        var_quantiles = Array{Float64}(undef, 3, var_length1, var_length2)
        for ii in 1:var_length1
            for t in 1:var_length2
                var_quantiles[:, ii, t] = quantile([obs[ii,t] for obs in var_cut], probs)
            end
        end

        # Estrai i quantili
        lower = var_quantiles[1, :, :]
        median = var_quantiles[2, :, :]
        upper = var_quantiles[3, :, :]
    end

    return Dict(:lower => lower, :median => median, :upper => upper)

end




function compare_estimates(g_wrap, gamma_wrap, f_wrap, b_wrap, dat, theta_true, k, row)
    println("Comparing estimates for row: ", row)

    # Extract original and estimated values
    g_true = dat[:g][row, k, :]
    f_true = dat[:f][k,:]
    beta_true = theta_true[k][:beta]  
    gamma_true = theta_true[k][:gamma]

    g_est = g_wrap[:median][row, :]
    g_lower = g_wrap[:lower][row, :]
    g_upper = g_wrap[:upper][row, :]

    gamma_est = gamma_wrap[:median]
    gamma_lower = gamma_wrap[:lower]
    gamma_upper = gamma_wrap[:upper]

    f_est = f_wrap[:median]
    f_lower = f_wrap[:lower]
    f_upper = f_wrap[:upper]

    beta_est = b_wrap[:median]
    beta_lower = b_wrap[:lower]
    beta_upper = b_wrap[:upper]

    # Print tables
    println("\nComparison for g (row $row):")
    println(DataFrame(Time=1:length(g_true), True=g_true, 
                      Estimate=g_est, Lower=g_lower, Upper=g_upper))

    println("\nComparison for f:")
    println(DataFrame(Time=1:length(f_true), True=f_true, 
                      Estimate=f_est, Lower=f_lower, Upper=f_upper))

    println("\nComparison for Beta:")
    println(DataFrame(Index=1:length(beta_true), True=beta_true, 
                      Estimate=beta_est, Lower=beta_lower, Upper=beta_upper))

    println("\nComparison for Gamma:")
    println(DataFrame(Index=1:length(gamma_true), True=gamma_true, 
                    Estimate=gamma_est, Lower=gamma_lower, Upper=gamma_upper))

    # Plot results
    p1 = plot(1:length(g_true), g_true, label="g True", lw=2)
    plot!(p1, 1:length(g_est), g_est, ribbon=(g_est .- g_lower, g_upper .- g_est), 
          label="g Estimate", lw=2, fillalpha=0.3)

    p2 = scatter(1:length(gamma_true), gamma_true, label="Gamma True", marker=:circle)
    scatter!(p2, 1:length(gamma_est), gamma_est, yerror=(gamma_est .- gamma_lower, gamma_upper .- gamma_est),
             label="Gamma Estimate")

    p3 = plot(1:length(f_true), f_true, label="f True", lw=2)
    plot!(p3, 1:length(f_est), f_est, ribbon=(f_est .- f_lower, f_upper .- f_est), 
          label="f Estimate", lw=2, fillalpha=0.3)

    p4 = scatter(1:length(beta_true), beta_true, label="Beta True", marker=:circle)
    scatter!(p4, 1:length(beta_est), beta_est, yerror=(beta_est .- beta_lower, beta_upper .- beta_est),
             label="Beta Estimate")

    plot(p1, p2, p3, p4, layout=(2,2), size=(900, 600)) #p2
end


function plot_acf_histogram(chain::Vector{T}, param_name::String) where T
    lags = collect(1001:5:length(chain))  # Generate indices 1, 6, 11, 16, ...
    acf_values = autocor(chain, lags)  # Compute ACF for selected lags

    scatter(1:length(lags), acf_values, xlabel="Lag", ylabel="Autocorrelation", 
           title="ACF Scatter Plot of $param_name", legend=false, markersize=5)
end


# in ingresso un vettore unitario, lanciare la funziona per K fissato
function latent_param_retrieval(f_true,f_wrap, gamma, gamma_wrap)
 
    f_norm = norm(f_true,2)
    f_hnorm = norm(f_wrap[:median],2)
 
    f_true_unit = f_true / f_norm
    gam_plus = gamma + log(f_norm)*ones(32)
 
    f_est = f_wrap[:median] / f_hnorm
    f_lower = f_wrap[:lower] / f_hnorm
    f_upper = f_wrap[:upper] / f_hnorm
 
    gamma_est = gamma_wrap[:median] + log(f_hnorm)*ones(32)
    gamma_lower = gamma_wrap[:lower] + log(f_hnorm)*ones(32)
    gamma_upper = gamma_wrap[:upper] + log(f_hnorm)*ones(32)
 
    # Plot results
    p1 = scatter(1:length(gam_plus), gam_plus, label="Gamma True Plus", marker=:circle)
    scatter!(p1, 1:length(gamma_est), gamma_est, yerror=(gamma_est .- gamma_lower, gamma_upper .- gamma_est),
                label="Gamma Estimate Plus")
 
    p2 = plot(1:length(f_true_unit), f_true_unit, label="f True Normalized", lw=2)
    plot!(p2, 1:length(f_est), f_est, ribbon=(f_est .- f_lower, f_upper .- f_est),
            label="f Estimate Normalized", lw=2, fillalpha=0.3)
           
    plot(p1, p2, layout=(2,1), size=(900, 600))
end