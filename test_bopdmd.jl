using DrWatson
quickactivate("/home/michael/Documents/Julia/DMD/env")
using LinearAlgebra, Plots
using Distributions, Random, Statistics
using DataFrames, Pickle, CSV, NPZ
using ProgressMeter

include("bopdmd.jl")

τ, d = 1, 10

df = CSV.read("/home/michael/Documents/Python/HAVOK/delase/data/all_human_data_metadata.csv", DataFrame)
df = select(df, Not(:Column1))

Xs = npzread("/home/michael/Documents/Python/HAVOK/delase/data/all_human_data.npy")
Xs = mapslices(x -> delayEmbed(x, τ, d), Xs, dims=(2, 3))
Xs = permutedims(Xs, (1, 3, 2))

dt = 0.01
t = dt .* (0:size(Xs, 3)-1)


B, eig, eig_std, _mode, mode_std, amplitude_std = fitBOPDMD(Xs[1, :, :], t, eig_constraints=String["stable"])

Bs = Vector{Union{Vector{ComplexF64}, Nothing}}(nothing, size(Xs, 1))
eigs = Vector{Union{Vector{ComplexF64}, Nothing}}(nothing, size(Xs, 1))
eigs_std = Vector{Union{Vector{Float64}, Nothing}}(nothing, size(Xs, 1))
modes = Vector{Union{Matrix{ComplexF64}, Nothing}}(nothing, size(Xs, 1))
modes_std = Vector{Union{Vector{Float64}, Nothing}}(nothing, size(Xs, 1))
amplitudes_std = Vector{Union{Vector{Float64}, Nothing}}(nothing, size(Xs, 1))

p = Progress(size(Xs, 1))
Threads.@threads for ii in axes(Xs, 1)
    B, eig, eig_std, _mode, mode_std, amplitude_std = fitBOPDMD(Xs[ii, :, :], t, eig_constraints=String["stable"])
    Bs[ii] = complex.(B)
    eigs[ii] = eig
    eigs_std[ii] = eig_std
    modes[ii] = _mode
    modes_std[ii] = mode_std[1, :]
    amplitudes_std[ii] = amplitude_std
    next!(p)
end
finish!(p)

bad_inds = findall(isnothing, Bs)
all_inds = 1:size(Xs, 1)
keep_inds = setdiff(all_inds, bad_inds)
valid_df = df[keep_inds, :]

valid_Bs = filter(!isnothing, Bs)
valid_eigs = filter(!isnothing, eigs)
valid_eigs_std = filter(!isnothing, eigs_std)
valid_modes = filter(!isnothing, _mode)
valid_modes_std = filter(!isnothing, modes_std)
valid_amplitudes_std = filter(!isnothing, amplitudes_std)

plot_color_mapping = Dict("AB" => :blue, "LF" => :green, "HF" => :orange)
# Plot eigenvalues
plot(real(valid_eigs[1]), imag(valid_eigs[1]), seriestype=:scatter, color=plot_color_mapping[valid_df[1, :lf_or_hf]], legend=false)
for ii in 2:size(valid_eigs, 1)
    plot!(real(valid_eigs[ii]), imag(valid_eigs[ii]), seriestype=:scatter, color=plot_color_mapping[valid_df[ii, :lf_or_hf]], legend=false)
end
# plot unit circle
phi = 0:0.01:2pi
scatter!(sin.(phi), cos.(phi), seriestype=:scatter, color=:black, legend=false, aspect_ratio=1)
xlims!(-2.0, 2.0)
ylims!(-2.0, 2.0)
savefig("figures/eigenvalues.pdf")