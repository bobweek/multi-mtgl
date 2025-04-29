using Integrals
using Cubature
using Distributions
using LinearAlgebra

d = 4

# number standard deviations around the mean to approximate Cov with
# higher numbers increase accuracy but decrease speed
# Infs lead to NaNs
nsdv = 4


r = rand(Exponential(1))
𝐛 = randn(d)
𝐛 = zeros(d)
𝛉 = randn(d)
𝚿 = rand(Wishart(2,Matrix(I,d,d)))
c = rand(Exponential(1))

𝐳 = randn(d)
𝐆 = rand(Wishart(2,Matrix(I,d,d)))
𝐄 = rand(Wishart(2,Matrix(I,d,d)))
n = rand(Exponential(100))

𝐏 = 𝐆+𝐄

# fitness function
function m(z, n, 𝐳, 𝐆, 𝐄, t)
    r + 𝐛'z - 0.5 * (𝛉-z)'𝚿 * (𝛉-z) - c*n
end

m(zeros(d), n, 𝐳, 𝐆, 𝐄, t)

# mean fitness
m̄ = r + 𝐛'𝐳 - 0.5 * (𝛉-𝐳)'𝚿 * (𝛉-𝐳) - 0.5 * tr(𝚿*𝐏) - c*n

m̄

# defines region over which to approximate integrals
lb = 𝐳 - nsdv*.√(diag(𝐏))
ub = 𝐳 + nsdv*.√(diag(𝐏))
# domain = (fill(-Inf,d), fill(Inf,d)) # returns NaN
domain = (lb,ub) # (lb, ub)


dist = MvNormal(𝐳,𝐏)
f = (x, q) -> m(x, n, 𝐳, 𝐆, 𝐄, 0)*pdf(dist, x)
prob = IntegralProblem(f, domain)

sol = solve(prob, CubatureJLp(), reltol = 1e-3, abstol = 1e-3).u

solve()

sol.retcode
sol

𝐳̇ = 𝐏*𝐛 + 𝐏*𝚿*(𝛉-𝐳)

function Covₘ(f,𝐳,𝐏,nsdv)

    # lower and upper bounds of domain
    lb = 𝐳 - nsdv*.√(diag(𝐏))
    ub = 𝐳 + nsdv*.√(diag(𝐏))
    # domain = (fill(-Inf,d), fill(Inf,d)) # returns NaN    

    # define integrand
    int = (x, p) -> (x-𝐳)*m(x)*pdf(dist, x)
    
    # define integral
    prob = IntegralProblem(int, (lb,ub))

    # compute integral (ie, estimate covariance)
    sol = solve(prob, CubatureJLp(), reltol = 1e-3, abstol = 1e-3)

    return sol.u

end

f = x -> (x-𝐳)

@time Covₘ(f,𝐳,𝐏,nsdv)

mz = (x, p) -> (x-𝐳)*m(x)*pdf(dist, x)
prob2 = IntegralProblem(mz, domain)
@time solve(prob2, CubatureJLp(), reltol = 1e-3, abstol = 1e-3)

𝐏̇ = -𝐏*𝚿*𝐏

# variance dynamics are more sensitive?
nsdv = 4
lb = 𝐳 - nsdv*.√(diag(𝐏))
ub = 𝐳 + nsdv*.√(diag(𝐏))
domain = (lb,ub) # (lb, ub)

#
# should redefine below using vech so no redundant calculations
#

function vech(A::Matrix{Float64})
    d = size(𝐏)[1]
    v = Vector{Float64}(undef, div(d*(d+1),2))
    k = 0
    for j = 1:d, i = j:d
        @inbounds v[k += 1] = A[i,j]
    end
    return v
end

vech(𝐏̇)

mzz = (x, p) -> pdf(dist, x)*m(x)*((x-𝐳)*(x-𝐳)' - 𝐏)
prob3 = IntegralProblem(mzz, domain)
@time solve(prob3, CubatureJLp(), reltol = 1e-3, abstol = 1e-3)

res.u

res = vech(𝐏̇)

k = 1
for j = 1:d, i = j:d

    mzz = (x, p) -> pdf(dist, x)*m(x)*((x[i]-𝐳[i])*(x[j]-𝐳[j]) - 𝐏[i,j])
    prob31 = IntegralProblem(mzz, domain)
    sol = solve(prob31, CubatureJLp(), reltol = 1e-3, abstol = 1e-3)
    res[k] = sol.u
    k += 1

end

i = 1
j = 2

mzz = (x, p) -> pdf(dist, x)*m(x)*((x[i]-𝐳[i])*(x[j]-𝐳[j]) - 𝐏[i,j])
prob31 = IntegralProblem(mzz, domain)
@time solve(prob31, CubatureJLp(), reltol = 1e-3, abstol = 1e-3)


Covₘ(mzz,domain)
