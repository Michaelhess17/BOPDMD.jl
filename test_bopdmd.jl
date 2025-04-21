using DrWatson
quickactivate("/home/michael/Documents/Julia/DMD/env")
using LinearAlgebra
using Plots
using Distributions, Random, Statistics
include("bopdmd.jl")

# Helper to flush output
flush_println() = flush(stdout)

function delayEmbed(X::AbstractMatrix{<:Number}, τ::Int, d::Int)
    n, m = size(X)
    if n == 0 || m == 0
        error("Input matrix X to delayEmbed is empty.")
    end
    num_delay_rows = n - (d - 1) * τ
    if num_delay_rows <= 0
        error("Cannot delay embed: result would have non-positive rows. n=$n, d=$d, τ=$τ")
    end
    X_delay = zeros(eltype(X), num_delay_rows, m * d)
    # Use explicit loops for clarity, @Threads.threads can be added back if needed
    for i in 1:num_delay_rows
        for j in 1:d
            start_row_idx = i + (j - 1) * τ
            start_col_idx = (j - 1) * m + 1
            end_col_idx = j * m
            # Ensure indices are within bounds
            if start_row_idx > n
                 error("Index out of bounds during delay embedding: start_row_idx=$start_row_idx > n=$n")
            end
            X_delay[i, start_col_idx:end_col_idx] = X[start_row_idx, :]
        end
    end
    return X_delay
end

# --- 1. Generate Test Data (Lorenz System with Noise) ---
function lorenz_step!(du, u, p, t)
    σ, ρ, β = p
    x, y, z = u
    du[1] = σ * (y - x)
    du[2] = x * (ρ - z) - y
    du[3] = x * y - β * z
end

function generate_lorenz_data(u0, p, dt, t_span)
    t = range(t_span[1], stop=t_span[2], step=dt)
    n_steps = length(t)
    dim = length(u0)
    X = zeros(Float64, dim, n_steps) # Use Float64 for consistency
    X[:, 1] = Float64.(u0)
    du = zeros(Float64, dim)

    for i in 1:(n_steps - 1)
        # Simple Euler step matching the original test script
        lorenz_step!(du, X[:, i], p, t[i])
        X[:, i+1] = X[:, i] + du * dt
    end
    return t, X
end

# Lorenz parameters
σ = 10.0
ρ = 28.0
β = 8.0 / 3.0
p = [σ, ρ, β]
u0 = [1.0, 0.0, 0.0]
dt = 0.01
t_span_train = (0.0, 10.0)
t_span_test = (10.0 + dt, 15.0) # Time range for forecasting

# Generate training data
t_train, X_train_clean = generate_lorenz_data(u0, p, dt, t_span_train)
println("Generated Clean Training Data: X shape ", size(X_train_clean), ", t shape ", size(t_train))
flush_println()

# Add noise
noise_level = 0.05 # 5% noise relative to standard deviation
Random.seed!(123)
noise_std = noise_level .* std(X_train_clean, dims=2)
X_train_noisy = X_train_clean .+ noise_std .* randn(size(X_train_clean))
println("Added Noise: Noisy Training Data shape ", size(X_train_noisy))
flush_println()

# Generate clean future data for comparison
# Use the last point of CLEAN training data to start test data
t_test, X_test_clean = generate_lorenz_data(X_train_clean[:, end], p, dt, t_span_test)
println("Generated Clean Test Data: X shape ", size(X_test_clean), ", t shape ", size(t_test))
flush_println()

# --- Transpose data: IMPORTANT - Julia scripts seem to expect time x features --- 
# (Based on original script's X_train_noisy', X_test_clean')
X_train_noisy_t = Matrix(X_train_noisy') # time x features
X_test_clean_t = Matrix(X_test_clean')   # time x features
println("Transposed Data: X_train_noisy_t shape ", size(X_train_noisy_t), ", X_test_clean_t shape ", size(X_test_clean_t))
flush_println()

# --- 1b. Delay Embed Data --- 
# Apply delay embedding *after* transposing
delay_d = 10
delay_tau = 1
X_train_delay = delayEmbed(X_train_noisy_t, delay_tau, delay_d)
X_test_delay = delayEmbed(X_test_clean_t, delay_tau, delay_d)

# Adjust time vectors for delay embedding
embedding_lag = (delay_d - 1) * delay_tau
t_train_delay = t_train[1:end-embedding_lag]
t_test_delay = t_test[1:end-embedding_lag]

println("Delay Embedded Data: Train shape ", size(X_train_delay), ", Test shape ", size(X_test_delay))
println("Delay Embedded Time: Train shape ", size(t_train_delay), ", Test shape ", size(t_test_delay))
flush_println()

# --- Use DELAYED data for fitting ---
X_fit_pre_scale = X_train_delay
t_fit = t_train_delay
X_compare_test_pre_scale = X_test_delay
t_compare_test = t_test_delay

# --- 1c. Center and Scale Data --- 
# Calculate mean and std based on training data (dim 1: time)
mean_X_fit = mean(X_fit_pre_scale, dims=1)
std_X_fit = std(X_fit_pre_scale, dims=1)
std_X_fit[std_X_fit .== 0] .= 1.0 # Avoid division by zero

# Apply scaling
X_fit_scaled = (X_fit_pre_scale .- mean_X_fit) ./ std_X_fit
# Apply training mean/std to test data
X_compare_test_scaled = (X_compare_test_pre_scale .- mean_X_fit) ./ std_X_fit

println("Scaled Data: Train shape ", size(X_fit_scaled), ", Test shape ", size(X_compare_test_scaled))
flush_println()

# --- Use SCALED data for final fit ---
X_final_fit = X_fit_scaled
X_final_test_compare = X_compare_test_scaled # This is the *scaled* version for comparison plot
t_final_test_compare = t_compare_test # Time vector remains the same for comparison

println("Using Scaled & Delayed Data for Fit: X_final_fit shape ", size(X_final_fit))
flush_println()

num_train_snapshots, num_features = size(X_final_fit) # time x features
println("Data Info: num_train_snapshots=$num_train_snapshots, num_features=$num_features")
flush_println()

# --- 2a. Set up BOPDMD Parameters --- 
svd_rank = 0 # Let BOPDMD find the rank (especially important with delay embedding)
_use_proj = true # Fit projected data
compute_A = false # Don't compute full A
verbose = true # Print progress from varpro
eig_constraints = ["stable"] # Example: constrain eigenvalues
_init_lambda::Float64 = 1e-6
maxlam::Int = 100
lamup::Float64 = 2.0
use_levmarq::Bool = true
maxiter::Int = 100 # Reduced maxiter for testing, increase later if needed
tol::Float64 = 1e-6
eps_stall::Float64 = 1e-10
use_fulljac::Bool = true
num_trials::Int = 50
trial_size::Float64 = 0.8 # Use 80% of data per trial
maxfail::Int = 100
remove_bad_bags::Bool = false

config = [svd_rank, _use_proj, compute_A, verbose, eig_constraints, _init_lambda, maxlam, lamup, use_levmarq, maxiter, tol, eps_stall, use_fulljac, num_trials, trial_size, maxfail, remove_bad_bags]

println("--- BOPDMD Configuration ---")
println("svd_rank: $svd_rank")
println("_use_proj: $_use_proj")
println("compute_A: $compute_A")
println("eig_constraints: $eig_constraints")
println("_init_lambda: $_init_lambda")
println("maxlam: $maxlam")
println("lamup: $lamup")
println("use_levmarq: $use_levmarq")
println("maxiter: $maxiter")
println("tol: $tol")
println("eps_stall: $eps_stall")
println("use_fulljac: $use_fulljac")
println("num_trials: $num_trials")
println("trial_size: $trial_size")
println("maxfail: $maxfail")
println("remove_bad_bags: $remove_bad_bags")
flush_println()

# --- 2. Run BOPDMD Pipeline ---
println("\n--- Starting BOPDMD Fit ---")
flush_println()
# IMPORTANT: fitBOPDMD expects features x time based on its internal transpose
# Provide the *transpose* of X_final_fit (features x time)
b, eigs, eigs_std, modes, modes_std, amps_std = fitBOPDMD(
    Matrix(X_final_fit'), # Pass features x time
    t_fit,
    config...
)
println("\n--- BOPDMD Fit Complete ---")
flush_println()

println("Modes shape (features x rank): ", size(modes))
println("Eigenvalues shape (rank): ", size(eigs))
println("Amplitudes shape (rank): ", size(b))
println("Eigenvalues std shape (rank): ", size(eigs_std))
println("Modes std shape (features x rank): ", size(modes_std))
println("Amplitudes std shape (rank): ", size(amps_std))
# Print first few elements carefully, check if complex
println("Modes (first 5x5, real part):\n", real.(modes[1:min(5, size(modes,1)), 1:min(5, size(modes,2))]))
println("Eigenvalues (first 5):\n", eigs[1:min(5, length(eigs))])
println("Amplitudes (first 5):\n", b[1:min(5, length(b))])
flush_println()

# --- 3. Reconstruct Training Data ---
println("\n--- Reconstructing Training Data ---")
flush_println()
# Use the mean parameters for reconstruction
t_train_row = reshape(t_fit, 1, length(t_fit))
# Ensure types are compatible for matrix multiplication
eigs_complex = ComplexF64.(eigs)
b_complex = ComplexF64.(b) # Amplitudes 'b' might need to be complex for dynamics
modes_complex = ComplexF64.(modes)

time_evolution_recon = exp.(eigs_complex * t_train_row)        # rank x time
dynamics_recon = diagm(b_complex) * time_evolution_recon       # rank x time
# Result should be features x time
X_reconstructed_fxt = modes_complex * dynamics_recon

# Transpose back to time x features for comparison and plotting
X_reconstructed_txf = Matrix(X_reconstructed_fxt')

println("Reconstruction shape (time x features): ", size(X_reconstructed_txf))
println("Reconstructed data (first 5 steps, all features):\n", real.(X_reconstructed_txf[1:min(5, size(X_reconstructed_txf, 1)), :]))
flush_println()

# --- 4. Forecast Future Data ---
println("\n--- Forecasting Future Data ---")
flush_println()
t_future = t_compare_test # Use the corresponding time vector for the test data

# Ensure correct feature count (from the original, non-delay-embedded data)
# This seems problematic if modes are based on delay-embedded features.
# Let's use the number of features from the modes matrix directly.
num_features_forecast = size(modes, 1) # Should be features * d

forecast_mean_fxt, forecast_var_fxt = forecast(
    t_future,
    num_trials, # Pass num_trials used in fit
    num_features_forecast, # Pass number of features in modes
    eigs, # Pass Complex eigs
    eigs_std, # Pass Float64 std
    b_complex, # Pass Complex amplitudes
    amps_std, # Pass Float64 std
    modes_complex # Pass Complex modes
)

# Transpose back to time x features for comparison and plotting
forecast_mean_txf = Matrix(forecast_mean_fxt')
forecast_var_txf = Matrix(forecast_var_fxt') # Variance is real

println("Forecast mean shape (time x features): ", size(forecast_mean_txf))
println("Forecast variance shape (time x features): ", size(forecast_var_txf))
println("Forecast mean (first 5 steps, all features):\n", real.(forecast_mean_txf[1:min(5, size(forecast_mean_txf, 1)), :]))

# Calculate standard deviation for plotting confidence intervals
# Use abs just in case of small numerical negatives in variance
forecast_std_dev_txf = sqrt.(abs.(forecast_var_txf))
flush_println()

# --- 5. Plotting ---
println("\n--- Generating Plots ---")
flush_println()

gr() # Ensure GR backend is active

# Define number of features to plot (original features, not delay-embedded)
num_original_features = size(X_train_noisy_t, 2) # From transposed noisy data
plot_indices = 1:num_original_features # Plot the first original feature dimensions

# Plot Reconstruction vs Training Data (Scaled & Delayed)
plt_recon = plot(layout=(num_original_features, 1), size=(800, 200 * num_original_features), legend=false)
for i in plot_indices
    plot!(plt_recon[i], t_fit, real.(X_final_fit[:, i]), label="Original Scaled Train (F$i)", linestyle=:dash)
    plot!(plt_recon[i], t_fit, real.(X_reconstructed_txf[:, i]), label="Reconstructed (F$i)")
    ylabel!(plt_recon[i], "Feature $i")
end
xlabel!(plt_recon[num_original_features], "Time")
title!(plt_recon[1], "BOPDMD Reconstruction vs Scaled Training Data (First Orig Feature Dim)")
plot!(plt_recon[1], legend=:outertopright) # Add legend to first plot
savefig(plt_recon, "jl_reconstruction.png")
println("Saved jl_reconstruction.png")
flush_println()

# Plot Forecast vs Test Data (Scaled & Delayed)
plt_forecast = plot(layout=(num_original_features, 1), size=(800, 200 * num_original_features), legend=false)
for i in plot_indices
    plot!(plt_forecast[i], t_compare_test, real.(X_final_test_compare[:, i]), label="True Scaled Future (F$i)", linestyle=:dash)
    plot!(plt_forecast[i], t_compare_test, real.(forecast_mean_txf[:, i]), label="Forecast Mean (F$i)")
    # Plot confidence interval
    lower_bound = real.(forecast_mean_txf[:, i]) .- 2 .* forecast_std_dev_txf[:, i]
    upper_bound = real.(forecast_mean_txf[:, i]) .+ 2 .* forecast_std_dev_txf[:, i]
    plot!(plt_forecast[i], t_compare_test, lower_bound, fillrange=upper_bound, fillalpha=0.3, label="±2 std dev", linealpha=0)
    ylabel!(plt_forecast[i], "Feature $i")
end
xlabel!(plt_forecast[num_original_features], "Time")
title!(plt_forecast[1], "BOPDMD Forecast vs Scaled Test Data (First Orig Feature Dim)")
plot!(plt_forecast[1], legend=:outertopright) # Add legend to first plot
savefig(plt_forecast, "jl_forecast.png")
println("Saved jl_forecast.png")
flush_println()

# Plot Eigenvalues
plt_eigs = scatter(real.(eigs), imag.(eigs), label="BOPDMD Eigs", mc=:blue, ms=5, msw=0, aspect_ratio=:equal)
# Add std dev circles
for (eig_val, std_val) in zip(eigs, eigs_std)
    # Plot 2 std dev circle (lighter)
    plot!(plt_eigs, Plots.partialcircle(0, 2π, 100, 2 * std_val), seriestype=[:shape], c=:blue, linecolor=:blue, fillalpha=0.2, aspect_ratio=:equal, label="", center=(real(eig_val), imag(eig_val)))
    # Plot 1 std dev circle (darker)
    plot!(plt_eigs, Plots.partialcircle(0, 2π, 100, std_val), seriestype=[:shape], c=:blue, linecolor=:blue, fillalpha=0.5, aspect_ratio=:equal, label="", center=(real(eig_val), imag(eig_val)))
end
# Re-plot points on top
scatter!(plt_eigs, real.(eigs), imag.(eigs), label="", mc=:blue, ms=3, msw=0)
plot!(plt_eigs, xlabel="Re(λ)", ylabel="Im(λ)", title="BOPDMD Eigenvalues", grid=true)
vline!(plt_eigs, [0], color=:black, lw=0.5, label="")
hline!(plt_eigs, [0], color=:black, lw=0.5, label="")
savefig(plt_eigs, "jl_eigenvalues.png")
println("Saved jl_eigenvalues.png")
flush_println()


println("\n--- Script Finished ---")
flush_println()