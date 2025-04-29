#
# this script contains functions for solving dn, d𝐳̄, and d𝐆 simultaneously
# by numerically estimating fitness gradients
#
# this script assumes m̄ has been obtained analytically
# and that m is independent of 𝐳̄ and 𝐆 so selection is
# frequency-independent
#   frequency dependent selection without analytical
#   expressions can be implemented using cov_fcts.jl
#

#
# (some) notation:
#   n = total abundance
#   𝐳 = d-dim trait value
#   𝐳̄ = mean trait vector
#   𝐆 = additive genetic covariance matrix
#   𝐄 = phenotypic noise covariance matrix
#   𝐏 = trait covariance matrix
#

using DifferentialEquations # methods for solving sde
using LinearAlgebra # for matrix square roots etc
using Distributions # for drawing parameters / initial conditions etc
using FiniteDiff # for estimating fitness gradients

# state variables n and 𝐳̄ stored as u = [n; 𝐳̄; 𝐆vec]
function unpack_state(u::AbstractVector)
    n = u[1]

    L = length(u)

    # Solve: total_len = 1 + d + d*(d+1)//2    
    discriminant = 9 + 8 * (L - 1)
    d = Int(round((-3 + sqrt(discriminant)) / 2))

    𝐳̄ = u[2 : d+1]

    dvec = div(d * (d + 1), 2)
    𝐆 = u[d+2 : d+1 + dvec]

    return n, 𝐳̄, 𝐌mat(𝐆)
end

# for structuring the noise process associated with random genetic drift
function noise_matrix(n::Number, z::Matrix{<:Number}, G::Matrix{<:Number})
    d1, d2 = size(z)
    @assert d1 == d2 "z must be a square matrix"
    d = d1

    k1, k2 = size(G)
    @assert k1 == k2 "G must be a square matrix"
    k = k1

    total_dim = 1 + d + k
    T = promote_type(typeof(n), eltype(z), eltype(G))
    M = zeros(T, total_dim, total_dim)

    M[1,1] = n
    M[2:1+d, 2:1+d] .= z
    M[2+d:end, 2+d:end] .= G

    return M
end

# vectorization of a symmetric matrix
function 𝐌vec(𝐌)
    d = size(𝐌)[1]
    M = zeros(floor(Int64, d * (d + 1) / 2))
    k = 1
    for i = 1:d
        for j = 1:i
            M[k] = 𝐌[i, j]
            k += 1
        end
    end
    return M
end

# symmetric matricization of a vector
function 𝐌mat(M)
    n = length(M)
    d = floor(Int, (sqrt(8n + 1) - 1) / 2)  # solve d(d+1)/2 = n
    𝐌 = zeros(d, d)
    k = 1
    for i = 1:d
        for j = 1:i
            𝐌[i, j] = M[k]
            𝐌[j, i] = M[k]
            k += 1
        end
    end
    return 𝐌
end

# for matricizing "symmetric" fourth-order tensors…
function vech_indices(d)
    return [(i,j) for j in 1:d for i in 1:j]
end

# matricizing 𝐒 = √𝚪
function 𝐒mat(𝐆)
    d = size(𝐆,1)
    idxs = vech_indices(d)
    k = length(idxs)
    S = zeros(k, k)
    
    for (a, (i,j)) in enumerate(idxs)
        E = zeros(d,d)
        E[i,j] = 1.0
        E[j,i] = 1.0
        B = E
        SB = (√𝐆 * B * √𝐆' + √𝐆 * B' * √𝐆')/√2
        for (b, (k,l)) in enumerate(idxs)
            S[b,a] = 0.5 * (SB[k,l] + SB[l,k])  # ensure symmetry
        end
    end
    return S
end

# deterministic components of stochastic differential equations
function f!(du, u, p, t) 
    (; 𝛍, v, 𝐄) = p

    n, 𝐳̄, 𝐆 = unpack_state(u)
    
    # approximate gradient of mean fitness wrt 𝐳̄
    m̄_𝐳̄ = z -> m̄(n,z,𝐆,𝐄,t)
    𝛃 = FiniteDiff.finite_difference_jacobian(m̄_𝐳̄,𝐳̄)'

    # approximate gradient of mean fitness wrt 𝐳̄
    m̄_𝐆 = G -> m̄(n,𝐳̄,G,𝐄,t)
    𝛄 = Symmetric(reshape(FiniteDiff.finite_difference_jacobian(m̄_𝐆,𝐆),d,d))

    # dynamics
    dn = m̄(n,𝐳̄,𝐆,𝐄,t)*n
    d𝐳̄ = 𝐆*𝛃
    d𝐆 = 𝛍 + 2*𝐆*𝛄*𝐆 - (v/n)*𝐆

    du[:] = [dn; d𝐳̄; 𝐌vec(d𝐆)]

end

# stochastic components of stochastic differential equations
function g!(du, u, p, t)
    (; 𝛍, v, 𝐄) = p

    n, 𝐳̄, 𝐆 = unpack_state(u)

    dn = √(v*n)
    d𝐳̄ = √(v/n) * √𝐆    
    d𝐆 = √(v/n) * 𝐒mat(𝐆)

    du[:,:] = noise_matrix(dn, d𝐳̄, d𝐆)

end

print("functions loaded!")
