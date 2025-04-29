#
# this script produces figure 4 of the main text
#

using StochasticDiffEq, Plots, Statistics, Printf

function f(ρ, δ, t)
    -δ * ρ * (1 - ρ^2) / 2
end

function g(ρ, δ, t)
    √δ * (1 - ρ^2)
end

T = 10^5
δ = 0.001
dt = 0.5
tspan = (0.0, T)
ϵ = 0.05
reps = 10
Δ = 100

Lrs = []

for i = 1:reps

    prob = SDEProblem(f, g, ρ₀, tspan, δ)
    sol = solve(prob, EM(), dt = dt)

    N = length(sol.t)
    Lr = []
    for t = 1:Δ:N
        pm1 = findall(x -> abs(x) > (1 - ϵ), sol.u[1:t])
        Lr = push!(Lr, length(pm1) / t)
    end

    Lrs = push!(Lrs, Lr)

    print("one down! ", reps - i, " more to go!\n")

end

Lrmn = mean(hcat(Lrs...), dims = 2)

plot(
    1:length(Lrmn),
    Lrmn,
    background_color = :transparent,
    background_color_inside = :transparent,
    labels = false,
    xlabel = "Time",
    ylabel = "Proportion of time spent near ±1",
)
