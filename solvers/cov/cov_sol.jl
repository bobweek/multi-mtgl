#
# this script sets up an example model and solves for n and 𝐳̄
# by estimating m̄ and Cov(m,𝐳) using numerical integration
#
#   estimating Cov(m,(𝐠-𝐳)(𝐠-𝐳)') is slow so 𝐆 is not solved for here
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

include("cov_fcts.jl") # loads functions for numerical solutions
using Plots # for plotting solutions

# trait dimensionality
d = 4

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

# drawing 𝐆 from Wishart distribution
df𝐆 = d+1
scale𝐆 = Matrix{Float64}(I, d, d)
𝐆 = rand(Wishart(df𝐆,scale𝐆),1)[1]
sqt𝐆 = √𝐆

# number of standard deviations away from mean
# for approximating integrals (m̄ and Cov(m,z))
nsdv = 4

#
# two options for optimum 𝛉 (fixed or oscillating)
#

# drawing stabilizing selection optimum from MVN
𝛉₀ = rand(MultivariateNormal(d, 1))

# oscillating optimum (to demonstrate how to incorporate dynamic model parameters)
function 𝛉(t)
    𝛉₀*sin(2*π*t)
end

# fitness function
#   for fixed optimum replace 𝛉(t) with 𝛉₀
function m(𝐳, n, 𝐳̄, 𝐆, 𝐄, t)
    r + 𝐛'𝐳 - 0.5 * (𝛉(t)-𝐳)'𝚿 * (𝛉(t)-𝐳) - c*n
end

#
# initial conditions
#

# drawing initial n from Gamma distribution
n₀ = rand(Gamma(2,10*v))

# drawing initial 𝐳 from MVN
𝐳̄₀ = rand(MultivariateNormal(d, 1))

# initial state variables
u₀ = [n₀; 𝐳̄₀]

# defining sde problem
p = (; v, 𝐄, 𝐆, sqt𝐆, nsdv, m) # parameters to pass to f and g
t_span = (0.0, 1.0)
noise_proto = noise_matrix(0,zeros(d,d))
sde_prob = SDEProblem(f!, g!, u₀, t_span, p, noise_rate_prototype = noise_proto)

# solving the system (and timing how long it takes with @time)
@time sol = solve(sde_prob, EM(), dt=1e-2, adaptive=false);

n_series = zeros(length(sol));
𝐳̄_series = zeros(length(sol),d);
Tᵢ = length(sol.t)
for i in 1:Tᵢ
    u = sol[i]
    n_series[i] = u[1]
    𝐳̄_series[i,:] .= u[2:end]
end

# plot solution
npl = plot(sol.t, n_series, xlabel=" ", ylabel="Abundance", label=false);
zpl = plot(sol.t, 𝐳̄_series, xlabel="Time", ylabel="Mean Traits", label=false);
pls = [npl zpl]
plot(pls...,layout=(2,1))