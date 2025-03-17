#
# a "DIY" implementation of the Euler-Maruyama algorithm
#     for numerically solving stochastic eco-evolutionary dynamics 
#
# currently only works for d ≤ 3 because of limitations in the Tensors package
#

using LinearAlgebra, Plots, PlotThemes, Tensors, Parameters, Statistics

#
# defining data structure and functions
#

@with_kw mutable struct PARS

    # simulation/integration parameters
    T::Int64 = 100  # time to integrate to
    N::Int64 = 2000 # number of time points
    # Δt = T/N

    # trait dimensionality
    d::Int64 = 2

    # initial conditions
    n₀::Float64 = 1e4
    𝐳₀::Vector{Float64} = zeros(d)
    𝐆₀::Matrix{Float64} = Matrix{Float64}(I, d, d)

    # phenotypic noise
    𝐄::Matrix{Float64} = Matrix{Float64}(I, d, d)

    # mutation matrix
    𝛍::Matrix{Float64} = 0.1 * Matrix{Float64}(I, d, d)

    # directional selection
    𝐛::Vector{Float64} = zeros(d)

    # stabilizing selection
    𝚿::Matrix{Float64} = Matrix{Float64}(I, d, d)

    # optimum
    𝛉::Vector{Float64} = zeros(d)

    # competition
    c::Float64 = 1e-4

    # intrinsic growth
    r::Float64 = 10.0

    # reproductive variance
    v::Float64 = 1.0

    # number of replicates
    R::Int64 = 10

end

function 𝐆vec(𝐆)

    d = size(𝐆)[1]

    G = zeros(floor(Int64, d * (d + 1) / 2))

    k = 1
    for i = 1:d
        for j = 1:i
            G[k] = 𝐆[i, j]
            k += 1
        end
    end

    return G

end

# evolves whole system in response to 
#   - mutation
#   - directional selection
#   - stabilizing selection
#   - demographic stochasticity
#   - random genetic drift
function evolve(p::PARS)
    @unpack_PARS p

    if d > 3
        print("can't have d > 3 because of the Tensors package :/\n")
        return
    end

    Δt = T / N
    N₁ = N + 1

    # setup history data structures
    nₕ = Vector{Float64}(undef, N₁)
    𝐳ₕ = Matrix{Float64}(undef, N₁, d)
    𝐆ₕ = Matrix{Float64}(undef, N₁, floor(Int64, d * (d + 1) / 2))

    # first entry is initial condition
    nₕ[1] = n₀
    𝐳ₕ[1, :] = 𝐳₀
    𝐆ₕ[1, :] = 𝐆vec(𝐆₀)

    # set state variables to initial conditions
    n = n₀
    𝐳 = 𝐳₀
    𝐆 = 𝐆₀

    # rate of drift
    δ = v / n

    # simulate/integrate
    for t = 2:N₁

        # phenotypic variance
        𝐏 = 𝐆 + 𝐄

        # abundance dynamics
        n +=
            (r + 𝐛'𝐳 - 0.5 * (𝛉 - 𝐳)'𝚿 * (𝛉 - 𝐳) - 0.5 * tr(𝚿 * 𝐏) - c * n) * n * Δt +
            √v * √n * √Δt * randn()

        # mean vector dynamics
        𝐳 += 𝐆 * (𝐛 + 𝚿 * (𝛉 - 𝐳)) * Δt + (√δ * √𝐆 * randn(d)) * √Δt

        # covariance tensor for genetic covariances
        Sqt𝐆 = Tensor{2,d}(√𝐆)
        𝚪 = Tensor{4,d}(otimesu(Sqt𝐆, Sqt𝐆) .+ otimesl(Sqt𝐆, Sqt𝐆)) / √2

        # a symmetric normal matrix
        𝐗 = √0.5 * Tensor{2,d}(randn(SymmetricTensor{2,d}) + diagm(randn(d)))

        # dynamics of 𝐆 ... 
        #            ⊡ is the double contraction
        #            have a look at https://ferrite-fem.github.io/Tensors.jl/stable/man/binary_operators/#Double-contraction
        𝐆 += Symmetric((𝛍 - 𝐆 * 𝚿 * 𝐆 - δ * 𝐆) * Δt + (√δ * 𝚪 ⊡ 𝐗) * √Δt)

        # append history
        𝐳ₕ[t, :] = 𝐳
        𝐆ₕ[t, :] = 𝐆vec(𝐆)

    end

    return 𝐳ₕ, 𝐆ₕ

end

# this evolves only 𝐆 in response to drift
function drift𝐆(p::PARS)
    @unpack_PARS p

    if d > 3
        print("can't have d > 3 because of the Tensors package :/\n")
        return
    end

    Δt = T / N
    N₁ = N + 1

    # setup history data structures
    𝐆ₕ = Matrix{Float64}(undef, N₁, floor(Int64, d * (d + 1) / 2))

    # first entry is initial condition
    𝐆ₕ[1, :] = 𝐆vec(𝐆₀)

    # set state variables to initial conditions
    𝐆 = 𝐆₀

    # rate of drift
    δ = v / n₀

    # simulate/integrate
    for t = 2:N₁

        # covariance tensor for trait covariances
        Sqt𝐆 = Tensor{2,d}(√𝐆)
        𝚪 = Tensor{4,d}(otimesu(Sqt𝐆, Sqt𝐆) .+ otimesl(Sqt𝐆, Sqt𝐆)) / √2

        # a symmetric normal matrix
        𝐗 = √0.5 * Tensor{2,d}(randn(SymmetricTensor{2,d}) + diagm(randn(d)))

        # dynamics of rxn norm vars
        𝐆 += Symmetric(-δ * 𝐆 * Δt .+ (√δ * √Δt) * (𝚪 ⊡ 𝐗))

        # append history
        𝐆ₕ[t, :] = 𝐆vec(𝐆)

    end

    return 𝐆ₕ

end

# run replicated expirements for whole system
function replicate(p::PARS)
    @unpack_PARS p

    𝐆dim = floor(Int64, d * (d + 1) / 2)
    𝐳 = Array{Float64}(undef, N + 1, d, R)
    𝐆 = Array{Float64}(undef, N + 1, 𝐆dim, R)

    for i = 1:R
        𝐳[:, :, i], 𝐆[:, :, i] = evolve(p)
    end

    return 𝐳, 𝐆

end

# run replicated experiments for just 𝐆
function 𝐆replicate(p::PARS)
    @unpack_PARS p

    𝐆dim = floor(Int64, d * (d + 1) / 2)
    𝐆 = Array{Float64}(undef, N + 1, 𝐆dim, R)

    for i = 1:R
        𝐆[:, :, i] = drift𝐆(p)
    end

    return 𝐆

end


# figure 1

p = PARS(n₀ = 10, 𝛍 = zeros(2, 2), T = 100, 𝐆₀ = [1.0 0.5; 0.5 1.0], N = 5000, R = 500);

ρ₀ˢ = [-0.5 0.0 0.5]

pls = []

for ρ₀ in ρ₀ˢ

    𝐆₀ = [1.0 ρ₀; ρ₀ 1.0]
    p.𝐆₀ = 𝐆₀

    # 𝐆replicating ... 
    𝐆 = 𝐆replicate(p)

    times = p.T * (0:p.N) / p.N
    ρ = 𝐆[:, 2, :] ./ sqrt.(𝐆[:, 1, :] .* 𝐆[:, 3, :])

    pl = plot(
        times,
        ρ,
        ylim = (-1, 1),
        background_color = :transparent,
        background_color_inside = :transparent,
        labels = false,
        linewidth = 1.0,
        linecolor = colorant"#3e8fb0",
        linealpha = 0.02,
        xlab = "Time",
        ylab = "Correlation",
    )

    push!(pls, pl)
end

plot(pls..., layout = (3, 1))


#
# figure 2
#

p = PARS(n₀ = 10, 𝛍 = zeros(2, 2), T = 100, 𝐆₀ = [1.0 0.5; 0.5 1.0], N = 5000, R = 500);

# 𝐆replicating ... 
𝐆 = 𝐆replicate(p)

# approximate expected dynamics
𝐆m = mean(𝐆, dims = 3)[:, :, 1]

# classical scaling result
Δt = p.T / p.N
δ = p.v / p.n₀
times = 0:Δt:p.T
Pt = p.𝐆₀[1, 1] .* exp.(-δ .* (0:Δt:p.T)) # diag
Qt = p.𝐆₀[1, 2] .* exp.(-δ .* (0:Δt:p.T)) # off-diag

theme(:bright)

plot(
    times,
    𝐆m[:, 1],
    label = "⟨G₁₁⟩",
    xlab = "Time",
    ylab = "Co/Variance",
    background_color = :transparent,
    background_color_inside = :transparent,
)
plot!(times, 𝐆m[:, 2], label = "⟨Ḡ₁₂⟩")
plot!(times, 𝐆m[:, 3], label = "⟨G₂₂⟩")
plot!(times, Pt, label = "𝔼G₁₁=𝔼G₂₂")
plot!(times, Qt, label = "𝔼G₁₂")
