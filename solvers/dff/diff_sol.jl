#
# this script  sets up an example model and solves dn, d𝐳̄, and d𝐆
# simultaneously by numerically estimating fitness gradients
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

include("dff_fcts.jl")
using Plots

# trait dimensionality
d = 4

# drawing mutation matrix from Wishart
df𝛍 = d +1
scale𝛍 = Matrix{Float64}(I, d, d)
𝛍 = rand(Wishart(df𝛍,scale𝛍))

# drawing trait noise matrix from Wishart
df𝐄 = d +1
scale𝐄 = 0.1*Matrix{Float64}(I, d, d)
𝐄 = rand(Wishart(df𝐄,scale𝐄))

#
# an example fitness function: m(𝐳)=r-𝐛'𝐳-(𝛉-𝐳)'𝚿(𝛉-𝐳)/2-cn
#

# intrinsic growth rate
r = 2

# drawing directional selection vector from MVN
𝐛 = 0.01*rand(MultivariateNormal(d,1))

# drawing stabilizing selection matrix from Wishart
df𝚿 = d +1
scale𝚿 = 0.001*Matrix{Float64}(I, d, d)
𝚿 = rand(Wishart(df𝚿,scale𝚿))

# strength of competition
c = 0.01

# reproductive variance
v = 1.0

#
# two options for optimum 𝛉 (fixed or oscillating)
#

# drawing stabilizing selection optimum from MVN
𝛉₀ = rand(MultivariateNormal(d, 1))

# oscillating optimum (to demonstrate how to incorporate dynamic model parameters)
function 𝛉(t)
    𝛉₀*sin(2*π*t)
end

# mean fitness
#   for fixed optimum replace 𝛉(t) with 𝛉₀
function m̄(n, 𝐳̄, 𝐆, 𝐄, t)
    𝐏 = 𝐆+𝐄
    r + 𝐛'𝐳̄ - 0.5 * (𝛉(t)-𝐳̄)'𝚿 * (𝛉(t)-𝐳̄) - 0.5 * tr(𝚿*𝐏) - c*n
end

#
# initial conditions
#

# drawing initial n from Gamma distribution
n₀ = rand(Gamma(2,10*v))

# drawing initial 𝐳 from MVN
𝐳̄₀ = rand(MultivariateNormal(d, 1))

# drawing initial 𝐆 from Wishart distribution
df𝐆 = d+1
scale𝐆 = Matrix{Float64}(I, d, d)
𝐆₀ = rand(Wishart(df𝐆,scale𝐆),1)[1]
𝐆₀vec = 𝐌vec(𝐆₀)
dvec = length(𝐆₀vec)

u₀ = [n₀; 𝐳̄₀; 𝐆₀vec]

#
# sde problem definition
#

p = (; 𝛍, v, 𝐄); # parameters to pass to f and g
t_span = (0.0, 10.0); # time span to solve until
noise_proto = noise_matrix(0,zeros(d,d),zeros(dvec,dvec)); # structuring noise process
sde_prob = SDEProblem(f!, g!, u₀, t_span, p, noise_rate_prototype = noise_proto)

#
# numerical solution
#   EM() ⟹ Euler-Marayama method to solve
#   with dt = 1e-3 time discretization
#
#   alternative methods are possible
#   see: https://docs.sciml.ai/DiffEqDocs/stable/solvers/sde_solve/#sde_solve
#

sol = solve(sde_prob, EM(), dt=1e-3, adaptive=false);

# checking solved 𝐆-matrix is positive-definite
Tᵢ = floor(Int,length(sol.t));
isposdef(𝐌mat(sol.u[Tᵢ][2+d:end]))

n_series = zeros(length(sol));
𝐳̄_series = zeros(length(sol),d);
𝐆_series = zeros(length(sol),dvec);

for i in 1:Tᵢ
    u = sol[i]
    n_series[i] = u[1]
    𝐳̄_series[i,:] .= u[2:1+d]
    𝐆_series[i,:] .= u[1+d+1:end]
end

# plot solutions
npl = plot(sol.t, n_series, xlabel=" ", ylabel="Abund\nances", label=false);
zpl = plot(sol.t, 𝐳̄_series, xlabel=" ", ylabel="Mean\nTraits", label=false);
Gpl = plot(sol.t, 𝐆_series, xlabel="Time", ylabel="Genetic\nCovariances", label=false);
pls = [npl zpl Gpl]
plot(pls...,layout=(3,1))
