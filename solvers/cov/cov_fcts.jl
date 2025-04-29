#
# this script contains functions for solving dn and d𝐳̄ simultaneously
# by estimating m̄ and Cov(m,𝐳) using numerical integration
#
#   estimating Cov(m,(𝐠-𝐳̄)(𝐠-𝐳̄)') is slow so 𝐆 is not solved for here
#   instead 𝐆 is assumed to be fixed
#
#   overall this approach is extremely slow and thus
#   should only be used when analytical expressions for
#   fitness gradients are not possible
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
using Distributions # to use trait distribution for m̄ and Cov(m,𝐳)
using Integrals # for numerical integration
using Cubature # for specific numerical integration method

# state variables n and 𝐳̄ stored as u = [n; 𝐳̄]
function unpack_state(u::AbstractVector)

    n = u[1]
    𝐳̄ = u[2:end]

    return n, 𝐳̄
end

# for structuring the noise process associated with random genetic drift
function noise_matrix(Mn::Number, M𝐳̄::Matrix{<:Number})
    d1, d2 = size(M𝐳̄)
    @assert d1 == d2 "M𝐳̄ must be a square matrix"
    d = d1

    total_dim = 1 + d
    T = promote_type(typeof(Mn), eltype(M𝐳̄))
    M = zeros(T, total_dim, total_dim)

    M[1,1] = Mn
    M[2:end, 2:end] .= M𝐳̄

    return M
end

# deterministic components of stochastic differential equations
function f!(du, u, p, t) 
    (; v, 𝐄, 𝐆, sqt𝐆, nsdv, m) = p

    n, 𝐳̄ = unpack_state(u)

    #
    # estimate m̄ and Cov(m,z)
    #

    # define domain to integrate over
    #   using Inf domain yields NaNs
    lb = 𝐳̄ - nsdv*.√(diag(𝐆+𝐄))
    ub = 𝐳̄ + nsdv*.√(diag(𝐆+𝐄))
    domain = (lb,ub)

    # trait distribution
    dist = MvNormal(𝐳̄,𝐆+𝐄)

    # integrand for m̄
    # q is a 'dummy' parameter needed for the integrator
    m̄int = (𝐳, q) -> m(𝐳, n, 𝐳̄, 𝐆, 𝐄, t)*pdf(dist, 𝐳)

    # defines integral for m̄
    m̄prob = IntegralProblem(m̄int, domain)

    # estimates m̄
    m̄ = solve(m̄prob, CubatureJLp(), reltol = 1e-3, abstol = 1e-3)

    # integrand for Cov(m,𝐳)
    Cint = (𝐳, q) -> (𝐳-𝐳̄)*m(𝐳, n, 𝐳̄, 𝐆, 𝐄, t)*pdf(p, 𝐳)

    # defines integral for Cov(m,𝐳)
    Cprob = IntegralProblem(Cint, domain)

    # estimates Cov(m,𝐳)
    Cov_m𝐳 = solve(Cprob, CubatureJLp(), reltol = 1e-3, abstol = 1e-3).u

    # dynamics    
    dn = m̄*n
    d𝐳̄ = Cov_m𝐳

    du[:] = [dn; d𝐳̄]

end

# stochastic components of stochastic differential equations
function g!(du, u, p, t)
    (; v, 𝐄, 𝐆, sqt𝐆, nsdv, m) = p

    n, 𝐳̄ = unpack_state(u)
    
    dn = √(v*n)
    d𝐳̄ = √(v/n)*sqt𝐆 # taking √𝐆 before solving saves time
    
    du[:,:] = noise_matrix(dn, d𝐳̄)

end

print("functions loaded!")