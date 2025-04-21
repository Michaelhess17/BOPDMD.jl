# Example usage of BOPDMD with Lorenz system data

import numpy as np
import torch
import matplotlib.pyplot as plt
from bopdmd import BOPDMD
import sys # For flushing print statements

# Helper function to flush print statements immediately
def flush_print():
    sys.stdout.flush()

# --- 1. Generate Test Data (Lorenz System with Noise) ---
def lorenz_step(u, p):
    """RK4 step for Lorenz system."""
    sigma, rho, beta = p
    x, y, z = u
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return torch.tensor([dxdt, dydt, dzdt])

def generate_lorenz_data(u0, p, dt, t_span):
    """Generate Lorenz data using RK4."""
    t = torch.arange(t_span[0], t_span[1] + dt, dt)
    n_steps = len(t)
    dim = len(u0)
    X = torch.zeros((dim, n_steps), dtype=torch.float64)
    X[:, 0] = torch.tensor(u0, dtype=torch.float64)

    for i in range(n_steps - 1):
        k1 = lorenz_step(X[:, i], p)
        k2 = lorenz_step(X[:, i] + 0.5 * dt * k1, p)
        k3 = lorenz_step(X[:, i] + 0.5 * dt * k2, p)
        k4 = lorenz_step(X[:, i] + dt * k3, p)
        X[:, i+1] = X[:, i] + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return t, X

# Lorenz parameters
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0
p = [sigma, rho, beta]
u0 = [1.0, 0.0, 0.0]
dt = 0.01
t_span_train = (0.0, 10.0)
t_span_test = (10.0 + dt, 15.0) # Time range for forecasting

# Generate training data
t_train, X_train_clean = generate_lorenz_data(u0, p, dt, t_span_train)
print(f"Generated Clean Training Data: X shape {X_train_clean.shape}, t shape {t_train.shape}")
flush_print()

# Add noise
noise_level = 0.05 # 5% noise relative to standard deviation
torch.manual_seed(123)
noise_std = noise_level * torch.std(X_train_clean, dim=1, keepdim=True)
X_train_noisy = X_train_clean + noise_std * torch.randn_like(X_train_clean)
print(f"Added Noise: Noisy Training Data shape {X_train_noisy.shape}")
flush_print()

# Generate clean future data for comparison
t_test, X_test_clean = generate_lorenz_data(X_train_clean[:, -1].numpy(), p, dt, t_span_test)
print(f"Generated Clean Test Data: X shape {X_test_clean.shape}, t shape {t_test.shape}")
flush_print()

# Transpose data to have time in rows (as expected by BOPDMD fit method)
X_train_noisy_t = X_train_noisy.T
X_test_clean_t = X_test_clean.T
print(f"Transposed Data: X_train_noisy_t shape {X_train_noisy_t.shape}, X_test_clean_t shape {X_test_clean_t.shape}")
flush_print()

# --- 1b. Delay Embed Data (Optional, adjust rank/params if used) ---
# Example:
# def delayEmbed(X, tau, d):
#     n, m = X.shape
#     X_delay = torch.zeros((n - (d - 1) * tau, m * d), dtype=X.dtype)
#     for i in range(n - (d - 1) * tau):
#         for j in range(d):
#             X_delay[i, j * m:(j + 1) * m] = X[i + j * tau, :]
#     return X_delay
#
# d = 10 # Example delay embedding dimension
# X_train_delay = delayEmbed(X_train_noisy_t, 1, d)
# X_test_delay = delayEmbed(X_test_clean_t, 1, d)
# t_train_delay = t_train[:-(d-1)]
# t_test_delay = t_test[:-(d-1)]
# print(f"Delay Embedded Data: Train shape {X_train_delay.shape}, Test shape {X_test_delay.shape}")
# print(f"Delay Embedded Time: Train shape {t_train_delay.shape}, Test shape {t_test_delay.shape}")
# flush_print()
# # Use delayed data for fitting:
# X_fit = X_train_delay
# t_fit = t_train_delay
# X_compare_test = X_test_delay
# t_compare_test = t_test_delay

# --- Use NON-DELAYED data for this example ---
X_fit = X_train_noisy_t
t_fit = t_train
X_compare_test = X_test_clean_t
t_compare_test = t_test
print(f"Using Non-Delayed Data for Fit: X_fit shape {X_fit.shape}, t_fit shape {t_fit.shape}")
flush_print()

# --- 1c. Center and Scale Data (Optional but recommended) ---
# # Calculate mean and std based on training data
# mean_X = torch.mean(X_fit, dim=0)
# std_X = torch.std(X_fit, dim=0)
# std_X[std_X == 0] = 1.0 # Avoid division by zero for constant features
#
# # Apply scaling
# X_fit_scaled = (X_fit - mean_X) / std_X
# X_compare_test_scaled = (X_compare_test - mean_X) / std_X # Use training mean/std
#
# print(f"Scaled Data: Train shape {X_fit_scaled.shape}, Test shape {X_compare_test_scaled.shape}")
# flush_print()
#
# # Use scaled data for fitting:
# X_final_fit = X_fit_scaled
# X_final_test_compare = X_compare_test_scaled

# --- Use NON-SCALED data for simplicity here ---
X_final_fit = X_fit
X_final_test_compare = X_compare_test
print(f"Using Non-Scaled Data for Fit: X_final_fit shape {X_final_fit.shape}")
flush_print()

num_features = X_final_fit.shape[1]
num_train_snapshots = X_final_fit.shape[0]
print(f"Data Info: num_features={num_features}, num_train_snapshots={num_train_snapshots}")
flush_print()

# --- 2a. Set up BOPDMD Parameters ---
# Note: svd_rank=0 finds optimal rank. If using delay embedding, might need larger rank.
svd_rank = 0  # Let BOPDMD find the rank
_use_proj = True # Fit projected data (more efficient)
compute_A = False # Don't compute full A (often large)
verbose = True # Print progress from varpro
eig_constraints = {"stable"} # Example: constrain eigenvalues to be stable
num_trials = 50 # Number of bagging trials
trial_size = 0.8 # Use 80% of data per trial
maxiter = 100 # Max iterations for varpro outer loop
tol = 1e-6   # Tolerance for varpro convergence

varpro_opts_dict = {
    "init_lambda": 1e-6,
    "maxlam": 100,
    "lamup": 2.0,
    "use_levmarq": True,
    "maxiter": maxiter,
    "tol": tol,
    "eps_stall": 1e-10,
    "use_fulljac": True,
    "verbose": verbose,
}

print("--- BOPDMD Configuration ---")
print(f"svd_rank: {svd_rank}")
print(f"_use_proj: {_use_proj}")
print(f"compute_A: {compute_A}")
print(f"eig_constraints: {eig_constraints}")
print(f"num_trials: {num_trials}")
print(f"trial_size: {trial_size}")
print(f"varpro_opts_dict: {varpro_opts_dict}")
flush_print()

# --- 2. Run BOPDMD Pipeline ---
print("\n--- Starting BOPDMD Fit ---")
flush_print()
bopdmd_model = BOPDMD(
    svd_rank=svd_rank,
    compute_A=compute_A,
    use_proj=_use_proj,
    num_trials=num_trials,
    trial_size=trial_size,
    eig_constraints=eig_constraints,
    varpro_opts_dict=varpro_opts_dict,
    remove_bad_bags=False # Keep all trial results for stats
)

bopdmd_model.fit(X_final_fit.numpy(), t_fit.numpy()) # BOPDMD expects numpy arrays currently
print("\n--- BOPDMD Fit Complete ---")
flush_print()

# Access results (convert back to torch if needed, results are numpy)
b = torch.from_numpy(bopdmd_model.amplitudes)
eigs = torch.from_numpy(bopdmd_model.eigs)
modes = torch.from_numpy(bopdmd_model.modes)
eigs_std = torch.from_numpy(bopdmd_model.eigenvalues_std) if bopdmd_model.eigenvalues_std is not None else None
modes_std = torch.from_numpy(bopdmd_model.modes_std) if bopdmd_model.modes_std is not None else None
amps_std = torch.from_numpy(bopdmd_model.amplitudes_std) if bopdmd_model.amplitudes_std is not None else None

print(f"Modes shape: {modes.shape}")
print(f"Eigenvalues shape: {eigs.shape}")
print(f"Amplitudes shape: {b.shape}")
if eigs_std is not None:
    print(f"Eigenvalues std shape: {eigs_std.shape}")
if modes_std is not None:
    print(f"Modes std shape: {modes_std.shape}")
if amps_std is not None:
    print(f"Amplitudes std shape: {amps_std.shape}")
print("Modes (first 5x5):", modes[:5, :5])
print("Eigenvalues (first 5):", eigs[:5])
print("Amplitudes (first 5):", b[:5])
flush_print()

# --- 3. Reconstruct Training Data ---
print("\n--- Reconstructing Training Data ---")
flush_print()
# Use the mean parameters for reconstruction
# Note: dynamics calculation uses numpy broadcasting rules
t_train_np = t_fit.numpy()
time_evolution_recon = np.exp(np.outer(eigs.numpy(), t_train_np)) # rank x time
dynamics_recon = np.diag(b.numpy()) @ time_evolution_recon       # rank x time
X_reconstructed_np = modes.numpy() @ dynamics_recon              # features x time
X_reconstructed = torch.from_numpy(X_reconstructed_np.T)         # time x features (to match X_final_fit)

print(f"Reconstruction shape (time x features): {X_reconstructed.shape}")
print("Reconstructed data (first 5 steps, all features):\n", X_reconstructed[:5, :])
flush_print()

# --- 4. Forecast Future Data ---
print("\n--- Forecasting Future Data ---")
flush_print()
t_future_np = t_compare_test.numpy()
# Forecast method returns mean and variance if num_trials > 0
forecast_result = bopdmd_model.forecast(t_future_np)

if isinstance(forecast_result, tuple):
    forecast_mean_np, forecast_var_np = forecast_result
    forecast_mean = torch.from_numpy(forecast_mean_np.T) # time x features
    forecast_var = torch.from_numpy(forecast_var_np.T)   # time x features
    print(f"Forecast mean shape (time x features): {forecast_mean.shape}")
    print(f"Forecast variance shape (time x features): {forecast_var.shape}")
    print("Forecast mean (first 5 steps, all features):\n", forecast_mean[:5, :])
    forecast_std_dev = torch.sqrt(torch.abs(forecast_var)) # Take abs for safety
else: # Deterministic forecast (num_trials <= 0)
    forecast_mean_np = forecast_result
    forecast_mean = torch.from_numpy(forecast_mean_np.T) # time x features
    forecast_var = None
    forecast_std_dev = None
    print(f"Forecast mean shape (time x features): {forecast_mean.shape}")
    print("Forecast mean (first 5 steps, all features):\n", forecast_mean[:5, :])

flush_print()


# --- 5. Plotting ---
print("\n--- Generating Plots ---")
flush_print()

plt.style.use('seaborn-v0_8-darkgrid')

# Plot Reconstruction vs Training Data
fig1, axs1 = plt.subplots(num_features, 1, figsize=(10, 2 * num_features), sharex=True)
if num_features == 1: axs1 = [axs1] # Ensure axs1 is iterable
for i in range(num_features):
    axs1[i].plot(t_fit, X_final_fit[:, i].real, label=f'Original Train (Feature {i+1})', linestyle=':')
    axs1[i].plot(t_fit, X_reconstructed[:, i].real, label=f'Reconstructed (Feature {i+1})')
    axs1[i].set_ylabel(f'Feature {i+1}')
    axs1[i].legend()
axs1[-1].set_xlabel('Time')
fig1.suptitle('BOPDMD Reconstruction vs Training Data')
plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to prevent title overlap
fig1.savefig("py_reconstruction.png")
print("Saved py_reconstruction.png")
flush_print()

# Plot Forecast vs Test Data
fig2, axs2 = plt.subplots(num_features, 1, figsize=(10, 2 * num_features), sharex=True)
if num_features == 1: axs2 = [axs2] # Ensure axs2 is iterable
for i in range(num_features):
    axs2[i].plot(t_compare_test, X_final_test_compare[:, i].real, label=f'True Future (Feature {i+1})', linestyle=':')
    axs2[i].plot(t_compare_test, forecast_mean[:, i].real, label=f'Forecast Mean (Feature {i+1})')
    if forecast_std_dev is not None:
        axs2[i].fill_between(t_compare_test,
                             (forecast_mean[:, i] - 2 * forecast_std_dev[:, i]).real,
                             (forecast_mean[:, i] + 2 * forecast_std_dev[:, i]).real,
                             alpha=0.3, label='±2 std dev')
    axs2[i].set_ylabel(f'Feature {i+1}')
    axs2[i].legend()
axs2[-1].set_xlabel('Time')
fig2.suptitle('BOPDMD Forecast vs True Future Data')
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
fig2.savefig("py_forecast.png")
print("Saved py_forecast.png")
flush_print()

# Plot Eigenvalues (requires BOPDMD model instance)
try:
    print("Plotting Eigenvalues...")
    flush_print()
    fig_eig, ax_eig = plt.subplots(figsize=(6, 6))
    bopdmd_model.plot_eig_uq(ax=ax_eig, draw_axes=True) # Pass ax to plot_eig_uq if it supports it
    # If plot_eig_uq doesn't take ax, plot manually:
    # if eigs_std is not None:
    #     for e, std in zip(eigs.numpy(), eigs_std.numpy()):
    #         c_1 = plt.Circle((e.real, e.imag), 2 * std, color="b", alpha=0.2)
    #         ax_eig.add_patch(c_1)
    #         c_2 = plt.Circle((e.real, e.imag), std, color="b", alpha=0.5)
    #         ax_eig.add_patch(c_2)
    # ax_eig.plot(eigs.real, eigs.imag, "o", c="b", label="BOP-DMD Eigs")
    # ax_eig.set_xlabel("Re(λ)")
    # ax_eig.set_ylabel("Im(λ)")
    # ax_eig.set_title("BOPDMD Eigenvalues")
    # ax_eig.grid(True)
    # ax_eig.axhline(0, color='black', lw=0.5)
    # ax_eig.axvline(0, color='black', lw=0.5)
    # ax_eig.legend()
    fig_eig.savefig("py_eigenvalues.png")
    print("Saved py_eigenvalues.png")
    flush_print()
except AttributeError as e:
    print(f"Could not plot eigenvalues using built-in method: {e}")
except Exception as e:
    print(f"An error occurred during eigenvalue plotting: {e}")


# Plot Modes (requires BOPDMD model instance and potentially reshaping)
# This is more complex as modes might be high-dimensional or delay-embedded
# For simple 1D/2D cases without delay, you could try:
# try:
#     print("Plotting Modes...")
#     flush_print()
#     num_modes_to_plot = min(5, modes.shape[1]) # Plot first few modes
#     # Assuming original data was 1D per feature * num_features
#     # If delay embedding was used, need to handle/reshape modes appropriately
#     feature_dim = modes.shape[0] # This might be feature_dim * d if delayed
#     # Simple plot assuming no delay embedding and modes are features x rank
#     fig_modes, axs_modes = plt.subplots(num_modes_to_plot, 1, figsize=(8, 2*num_modes_to_plot), sharex=True)
#     if num_modes_to_plot == 1: axs_modes = [axs_modes]
#     x_axis = np.arange(feature_dim) # Placeholder spatial axis
#     for i in range(num_modes_to_plot):
#         axs_modes[i].plot(x_axis, modes[:, i].real, label=f'Mode {i+1} (Real)')
#         if modes_std is not None:
#              axs_modes[i].fill_between(x_axis,
#                                        (modes[:, i] - 2 * modes_std[:, i]).real,
#                                        (modes[:, i] + 2 * modes_std[:, i]).real,
#                                        alpha=0.3, label='±2 std dev')
#         axs_modes[i].set_ylabel(f'Mode {i+1}')
#         axs_modes[i].legend()
#     axs_modes[-1].set_xlabel('Feature Index (Spatial Dim)')
#     fig_modes.suptitle('BOPDMD Modes')
#     plt.tight_layout(rect=[0, 0.03, 1, 0.97])
#     fig_modes.savefig("py_modes.png")
#     print("Saved py_modes.png")
#     flush_print()

# except Exception as e:
#     print(f"An error occurred during mode plotting: {e}")


print("\n--- Script Finished ---")
flush_print()
# Keep plots showing if not run in interactive mode?
# plt.show()
