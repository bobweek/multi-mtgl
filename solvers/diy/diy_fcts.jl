#
# this script contains functions for solving dn and d𝐳̄ simultaneously
# using a manual implementation of the Euler-Maruyama algorithm
#
# this was the original numerical implementation
# and is kept here because it is used to produce
# figures for the supplement
#
# it may also be useful as a pedagogical tool to learn
# about numerical implementations
#
# the implementations found in the scripts "knw_sol.jl" and "diff_sol.jl"
# are probably much faster and more reliable than this approach
#
# currently only works for d ≤ 3 because of limitations in the "Tensors.jl" package
#

using LinearAlgebra, Tensors, Parameters, Statistics

# data structure that carries simulation parameters
@with_kw mutable struct PARS

    # simulation/integration parameters
    T::Int64 = 100  # time to integrate to
    N::Int64 = 2000 # number of time points

    # trait dimensionality
    d::Int64 = 2

    # initial conditions
    n₀::Float64 = 1e4
    𝐳̄₀::Vector{Float64} = zeros(d)
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

# vectorize the 𝐆-matrix
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

# convert vectorized 𝐆 to matrix 𝐆
function 𝐆mat(𝐆vec)
    n = length(𝐆vec)
    d = floor(Int, (sqrt(8n + 1) - 1) / 2)  # solve d(d+1)/2 = n

    𝐆 = zeros(d, d)

    k = 1
    for i = 1:d
        for j = 1:i
            𝐆[i, j] = 𝐆vec[k]
            𝐆[j, i] = 𝐆vec[k]  # maintain symmetry
            k += 1
        end
    end

    return 𝐆
end

# evolves whole system (n,𝐳̄,𝐆) in response to 
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
    𝐳̄ₕ = Matrix{Float64}(undef, N₁, d)
    𝐆ₕ = Matrix{Float64}(undef, N₁, floor(Int64, d * (d + 1) / 2))

    # first entry is initial condition
    nₕ[1] = n₀
    𝐳̄ₕ[1, :] = 𝐳̄₀
    𝐆ₕ[1, :] = 𝐆vec(𝐆₀)

    # set state variables to initial conditions
    n = n₀
    𝐳̄ = 𝐳̄₀
    𝐆 = 𝐆₀

    # rate of drift
    δ = v / n

    # simulate/integrate
    for t = 2:N₁

        # phenotypic variance
        𝐏 = 𝐆 + 𝐄

        # abundance dynamics
        n +=
            (r + 𝐛'𝐳̄ - 0.5 * (𝛉 - 𝐳̄)'𝚿 * (𝛉 - 𝐳̄) - 0.5 * tr(𝚿 * 𝐏) - c * n) * n * Δt +
            √v * √n * √Δt * randn()

        # mean vector dynamics
        𝐳̄ += 𝐆 * (𝐛 + 𝚿 * (𝛉 - 𝐳̄)) * Δt + (√δ * √𝐆 * randn(d)) * √Δt

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
        nₕ[t] = n
        𝐳̄ₕ[t, :] = 𝐳̄
        𝐆ₕ[t, :] = 𝐆vec(𝐆)

    end

    return nₕ, 𝐳̄ₕ, 𝐆ₕ

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
    n = Array{Float64}(undef, N + 1, R)
    𝐳̄ = Array{Float64}(undef, N + 1, d, R)
    𝐆 = Array{Float64}(undef, N + 1, 𝐆dim, R)

    for i = 1:R
        n[:, i], 𝐳̄[:, :, i], 𝐆[:, :, i] = evolve(p)
    end

    return n, 𝐳̄, 𝐆

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

print("functions loaded!")