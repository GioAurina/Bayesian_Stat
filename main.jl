using Random, Plots, DataFrames, StatsBase, ToeplitzMatrices, CSV, HDF5

include("utilities.jl")
include("fit_MCMC.jl")
include("priors.jl")
include("proposal_functions.jl")
include("sampling_functions.jl")
include("target_functions.jl")
include("simulated_data.jl")


#function main()
seed = 1997

df = CSV.read("./FinalStations.csv", DataFrame; delim=',')
df.Elevation.=zscore(df.Elevation)

# Parameters
K = 2 #number of sources
n = 32 #number of sites
n_time = 30 #365

# generating simulated data
sites, dat, theta_true, dat_trials = simulate_data(df, seed, K, n, n_time)

####################################################################################

# Hyperparameters for MCMC
hyperparam = Dict(
    :tau_prior_sd => sqrt(3), 
    :tau_proposal_sd => 0.02,
    :rho_prior_shape => 0.02, 
    :rho_prior_scale => 1,
    :rho_proposal_sd => 0.05, 
    #:beta_prior_mu => 0, 
    #:beta_prior_sd => 1,
    :phi_prior_shape => 0.02,
    :phi_prior_scale => 1.0,
    :phi_proposal_sd => 0.01,
    :beta_proposal_sd => 0.1,
    :gamma_proposal_sd => 0.02
)

#theta0 = Dict{Int64, Dict{Any,Any}}()
theta0 = Dict(
    :rho => 0.3,
    :phi => 1/500.0,
    :beta => ones(4),
    :gamma => (ones(n)), 
    :tau => zeros(n) 
)


# Iterations of MCMC
k = 1 #fix a source
n_iter = 6000

Random.seed!(seed)
results = fit_model(sites, dat[:g][:,k,:], n_iter, theta0, hyperparam)
burn_in = 1000

# Save all the matrices in a file HDF5
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
        file["gamma_$i"] = results[:chain_gamma][i]
        file["tau_$i"] = results[:chain_tau][i]
        file["rho_$i"] = results[:chain_rho][i]
        file["g_hat_$i"] = results[:chain_g][i]
        file["z_$i"] = results[:chain_z][i]
        file["beta_$i"] = results[:chain_beta][i]
        file["phi_$i"] = results[:chain_phi][i]
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
latent_param_retrieval(dat[:f][k,:],f_wrap, dat[:gamma][:,k], gamma_wrap)

#Trace Plots

p13 = plot()
plot!(1:n_iter, [results[:chain_g][i][2,7] for i in 1:n_iter], label="g[2,5] Trace")
display(p13)

p14 = plot()
plot!(1:n_iter, [results[:chain_beta][i][4] for i in 1:n_iter], label="Beta[4] Trace")
display(p14)

p15 = plot()
plot!(1:n_iter, [results[:chain_gamma][i][4] for i in 1:n_iter], label="Gamma[4] Trace")
display(p15)

p16 = plot()
plot!(1:n_iter, [results[:chain_tau][i][4] for i in 1:n_iter], label="Tau[4] Trace")
display(p16)

p17 = plot()
plot!(1:n_iter, [results[:chain_rho][i] for i in 1:n_iter], label="Rho Trace")
plot!(1:n_iter, 0.2 * ones(n_iter))
display(p17)

p18 = plot()
plot!(1:n_iter, [results[:chain_phi][i] for i in 1:n_iter], label="Phi Trace")
plot!(1:n_iter, (1/400) * ones(n_iter))
display(p18)

# Compute ACF for g[2,4]
plot_acf_histogram([results[:chain_g][i][4,4] for i in burn_in:n_iter], "g[2,4]")

# Compute ACF for Beta[4]
plot_acf_histogram([results[:chain_beta][i][4] for i in burn_in+1:n_iter], "Beta[4]", burn_in)

# Compute ACF for Gamma[4]
plot_acf_histogram([results[:chain_gamma][i][4] for i in burn_in+1:n_iter], "Gamma[4]", burn_in)

# Compute ACF for Tau[4]
plot_acf_histogram([results[:chain_tau][i][4] for i in burn_in+1:n_iter], "Tau[4]", burn_in)

# Compute ACF for Rho
plot_acf_histogram([results[:chain_rho][i] for i in burn_in+1:n_iter], "Rho", burn_in)

# Compute ACF for Phi
plot_acf_histogram([results[:chain_phi][i] for i in burn_in+1:n_iter], "Phi", burn_in)

#=
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
=#

#######SOLO SE VOGLIO LEGGERE DA MATRICE H5


# Initialize results dictionary properly
    results = Dict(
        :chain_f => [],
        :chain_g => [],
        :chain_z => [],
        :chain_beta => Dict(),
        :chain_gamma => Dict(),
        :chain_tau => Dict(),
        :chain_phi => Dict(),
        :chain_rho => Dict(),
        :runtime => 0.0 # Placeholder for runtime if needed
    )
    
    h5open("matrici.h5", "r") do file
        # Read the true parameter values
        theta_true[k][:rho] = read(file["rho_true"])   
        theta_true[k][:phi] = read(file["phi_true"]) 
        theta_true[k][:beta] = read(file["beta_true"]) 
        theta_true[k][:tau] = read(file["tau_true"])  
        dat[:gamma] = read(file["gamma_true"]) 
        dat[:g] = read(file["g_true"]) 
        dat[:f] = read(file["f_true"]) 
    
        # Read MCMC chains
        for i in 1:n_iter
            push!(results[:chain_f], read(file["f_$i"]))  
            push!(results[:chain_g], read(file["g_hat_$i"]))
            push!(results[:chain_z], read(file["z_$i"]))
    
            # Store dictionaries for parameter chains
            results[:chain_beta][i] = read(file["beta_$i"])
            results[:chain_gamma][i] = read(file["gamma_$i"])
            results[:chain_tau][i] = read(file["tau_$i"])
            results[:chain_phi][i] = read(file["phi_$i"])
            results[:chain_rho][i] = read(file["rho_$i"])
        end
    end



p14 = plot()
plot!(burn_in:n_iter, [results[:chain_beta][i][4] for i in burn_in:n_iter], label="Beta[4] Trace")
display(p14)

p15 = plot()
plot!(burn_in:n_iter, [results[:chain_gamma][i][4] for i in burn_in:n_iter], label="Gamma[4] Trace")
display(p15)

p16 = plot()
plot!(burn_in:n_iter, [results[:chain_tau][i][4] for i in burn_in:n_iter], label="Tau[4] Trace")
display(p16)

p17 = plot()
plot!(burn_in:n_iter, [results[:chain_rho][i] for i in burn_in:n_iter], label="Rho Trace")  # Fixed indexing
display(p17)

p18 = plot()
plot!(burn_in:n_iter, [results[:chain_phi][i] for i in burn_in:n_iter], label="Phi Trace")  # Fixed indexing
display(p18)

