#
# this script sets up an example model and solves for n and 𝐳̄
# by estimating m̄ and Cov(m,z) using numerical integration
#
#   estimating Cov(m,(g-𝐳)(g-𝐳)') is slow so 𝐆 is not solved for here
#   instead 𝐆 is assumed to be fixed
#
#   overall this approach is extremely slow and thus
#   should only be used when analytical expressions for
#   fitness gradients are not possible
#

#
# (some) notation:
#   n = total abundance
#   z = d-dim trait value
#   𝐳̄ = mean trait vector
#   𝐆 = additive genetic covariance matrix
#   𝐄 = phenotypic noise covariance matrix
#   𝐏 = trait covariance matrix
#

include("knw_fcts.jl")
using Plots

# trait dimensionality
d = 4;

# drawing mutation matrix from Wishart
df𝛍 = d +1;
scale𝛍 = Matrix{Float64}(I, d, d);
𝛍 = rand(Wishart(df𝛍,scale𝛍))

# drawing trait noise matrix from Wishart
df𝐄 = d +1;
scale𝐄 = 0.1*Matrix{Float64}(I, d, d);
𝐄 = rand(Wishart(df𝐄,scale𝐄))

#
# an example fitness function: m(𝐳)=r-𝐛'𝐳-(𝛉-𝐳)'𝚿(𝛉-𝐳)/2-cn
#

# intrinsic growth rate
r = 2

# drawing directional selection vector from MVN
𝐛 = 0.01*rand(MultivariateNormal(d,1))

# drawing stabilizing selection matrix from Wishart
df𝚿 = d +1;
scale𝚿 = 0.001*Matrix{Float64}(I, d, d);
𝚿 = rand(Wishart(df𝚿,scale𝚿))

# strength of competition
c = 0.01;

# reproductive variance
v = 1.0;

# drawing stabilizing selection optimum from MVN
𝛉₀ = 10*rand(MultivariateNormal(d, 1))

# oscillating optimum (to demonstrate how to incorporate dynamic model parameters)
function 𝛉(t)
    𝛉₀*sin(4*π*t)
end

# mean fitness
#   for fixed optimum replace 𝛉(t) with 𝛉₀
function m̄(n, 𝐳̄, 𝐆, 𝐄, t)
    𝐏 = 𝐆+𝐄
    r + 𝐛'𝐳̄ - 0.5 * (𝛉(t)-𝐳̄)'𝚿 * (𝛉(t)-𝐳̄) - 0.5 * tr(𝚿*𝐏) - c*n
end

# gradient of mean fitness m̄ wrt 𝐳̄
#   for fixed optimum replace 𝛉(t) with 𝛉₀
function ∇𝐳̄m̄(n, 𝐳̄, 𝐆, 𝐄, t)
    𝐛 + 𝚿*(𝛉(t)-𝐳̄)
end

# gradient of mean fitness m̄ wrt 𝐆
function ∇𝐆m̄(n, 𝐳̄, 𝐆, 𝐄, t)
    -0.5*𝚿
end

# mean of fitness gradient wrt 𝐳̄
#   ol_∇𝐳̄m = ∫ ∇𝐳̄[m(𝐳)] p(𝐳)d𝐳
function ol_∇𝐳̄m(n, 𝐳̄, 𝐆, 𝐄, t)
    zeros(d)
end

# mean of fitness gradient wrt 𝐆
#   ol_∇𝐳̄m = ∫ ∇𝐆[m(𝐳)] p(𝐳)d𝐳
function ol_∇𝐆m(n, 𝐳̄, 𝐆, 𝐄, t)
    zeros(d,d)
end

#
# initial conditions
#

# drawing initial n from Gamma distribution
n₀ = rand(Gamma(2,10*v))

# drawing initial 𝐳 from MVN
𝐳̄₀ = rand(MultivariateNormal(d, 1))

# drawing initial 𝐆 from Wishart distribution
df𝐆 = d+1;
scale𝐆 = Matrix{Float64}(I, d, d);
𝐆₀ = rand(Wishart(df𝐆,scale𝐆),1)[1]
𝐆₀vec = 𝐌vec(𝐆₀);
dvec = length(𝐆₀vec);

u₀ = [n₀; 𝐳̄₀; 𝐆₀vec];

#
# sde problem definition
#

p = (; 𝛍, v, 𝐄); # parameters to pass to f and g
t_span = (0.0, 10.0); # time span to solve until
noise_proto = noise_matrix(0,zeros(d,d),zeros(dvec,dvec)); # structuring noise process
sde_prob = SDEProblem(f!, g!, u₀, t_span, p, noise_rate_prototype = noise_proto); # defines problem

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