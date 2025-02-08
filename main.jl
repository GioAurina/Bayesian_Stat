using Random, Plots, DataFrames, StatsBase, ToeplitzMatrices, CSV, HDF5

include("utilities.jl")
include("fit_MCMC.jl")
include("priors.jl")
include("proposal_functions.jl")
include("sampling_functions.jl")
include("target_functions.jl")
include("simulated_data.jl")


#function main()
    # Set a fixed seed
    seed = 1997

    df = CSV.read("./FinalStations.csv", DataFrame; delim=',')
    df.Elevation.=zscore(df.Elevation)
    
    # Parametri
    K = 2 #numero di fonti
    n = 32 #numero di siti
    n_time = 180 #365
    
    # generating simulated data
    sites, dat, theta_true, dat_trials = simulate_data(df, seed, K, n, n_time)

    ####################################################################################

    # Iperparametri per MCMC
    hyperparam = Dict(
        :tau_prior_sd => sqrt(3), 
        :tau_proposal_sd => 0.01,
        :rho_prior_shape => 0.02, 
        :rho_prior_scale => 1,
        :rho_proposal_sd => 0.01, 
        #:beta_prior_mu => 0, 
        #:beta_prior_sd => 1,
        :phi_prior_shape => 0.02,
        :phi_prior_scale => 1.0,
        :phi_proposal_sd => 0.05,
        :beta_proposal_sd => 0.1,
        :gamma_proposal_sd => 0.01
    )
    
    #theta0 = Dict{Int64, Dict{Any,Any}}()
    theta0 = Dict(
        :rho => 0.3,
        :phi => 1/500.0,
        :beta => ones(4),
        :gamma => (ones(n)), 
        :tau => zeros(n) 
    )


   
    # Punto e valore fissati (pinned point/value)
    #pinned_point = div(n_time, 2)  # punto fissato (metà del tempo)
   # pinned_value = mean(dat[:g][:, 1, pinned_point]) # valore medio della colonna `pinned_point` (in R 'apply(dat$y, 1, mean)[pinned_point]')

    # Iterazioni di MCMC
    k = 2
    n_iter = 15000

    Random.seed!(seed)
    results = fit_model(sites, dat[:g][:,k,:], n_iter, theta0, hyperparam)

    # Funzione per riassumere i risultati MCMC
    burn_in = 1000  # Calcolare il burn-in (primo 60%)

     # Salvare tutte le matrici in un unico file HDF5
     h5open("matrici.h5", "w") do file
        file["rho_true"] = theta_true[k][:rho]
        file["phi_true"] = theta_true[k][:phi]
        file["beta_true"] = theta_true[k][:beta]
        file["tau_true"] = theta_true[k][:tau]
        file["gamma_true"] = dat[:gamma]
        file["g_true"] = dat[:g]
        file["f_true"] = dat[:f]
        for i in 1:n_iter
            file["f_$i"] = results[:chain_f][i]
            file["gamma_$i"] = results[:chain][i][:gamma]
            file["tau_$i"] = results[:chain][i][:tau]
            file["rho_$i"] = results[:chain][i][:rho]
            file["g_hat_$i"] = results[:chain_g][i]
            file["z_$i"] = results[:chain_z][i]
            file["beta_$i"] = results[:chain][i][:beta]
            file["phi_$i"] = results[:chain][i][:phi]
        end
    end

    out_sim, mean_f, g_hat_quantiles = getSummaryOutput(results, dat_trials[k], dat[:g][:,k,:], burn_in)

     ####### PROVA DI WRAP dati

    ## try plot search only for median and quantiles
    #results = Dict(
    #    :chain => chain,
    #    :chain_f => chain_f,
    #    :chain_g => chain_g,
    #    :chain_z => chain_z,
    #    :chain_beta => chain_beta,
    #    :runtime => runtime
    #)
    # ultima voce è se la variabile è 2d o no
    g_wrap = wrap_data(results[:chain_g], burn_in, n_iter, false) 
    z_wrap = wrap_data(results[:chain_z], burn_in, n_iter, false) 
    f_wrap = wrap_data(results[:chain_f], burn_in, n_iter, true) 
    b_wrap = wrap_data(results[:chain_beta], burn_in, n_iter, true)
    gamma_wrap = wrap_data(results[:chain_gamma] , burn_in, n_iter, true) 

    compare_estimates(g_wrap, gamma_wrap, f_wrap, b_wrap, dat, theta_true, k, 1) # scegliere quale sito

# Ensure you use the correct index for `dat_trials[k]`
dat_trials_summary = combine(groupby(dat_trials[k], :time), :value => mean => :value_mean)

# Compute the median of predicted values
out_sim_summary = combine(groupby(out_sim, :time), :med => median => :med_median)

# Compute Mean Squared Error (MSE) for RPAGP
MSE_RPAGP = sum((out_sim_summary.med_median .- dat[:f][k, :]).^2) / n_time
println("MSE(RPAGP): ", MSE_RPAGP)

# Compute MSE for EMP
MSE_EMP = sum((dat_trials_summary.value_mean .- dat[:f][k, :]).^2) / n_time
println("MSE(EMP): ", MSE_EMP)

# Initialize plot
p12 = plot()

# Observed Data Bounds
plot!(p12, dat_trials[k].time, out_sim.lwr, label="Observed Data Lower", alpha=0.25, linewidth=1)
plot!(p12, dat_trials[k].time, out_sim.upr, label="Observed Data Upper", alpha=0.25, linewidth=1)

# Estimated f and True f
plot!(p12, out_sim_summary.time, out_sim_summary.med_median, label="Estimated f", linewidth=2, color=:chartreuse)
plot!(p12, out_sim_summary.time, dat[:f][k, :], label="True f", linestyle=:dash, linewidth=2, color=:red)

# Empirical Mean
plot!(p12, dat_trials_summary.time, dat_trials_summary.value_mean, label="Empirical Mean", linestyle=:dot, linewidth=2, color=:black)

display(p12)

# Beta Trace Plot
p14 = plot()
plot!(1:n_iter, [results[:chain][i][:beta][4] for i in 1:n_iter], label="Beta[4] Trace")
display(p14)
