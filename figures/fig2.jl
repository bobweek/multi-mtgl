#
# this script produces figure 2 of the main text
#

using StochasticDiffEq, Plots, Statistics, Printf

function f(ρ, δ, t)
    -δ * ρ * (1 - ρ^2) / 2
end

function g(ρ, δ, t)
    √δ * (1 - ρ^2)
end

ρ₀ = 0.0
δ = 0.001
dt = 0.1
tspan = (0.0, 10.0^4)

prob = SDEProblem(f, g, ρ₀, tspan, δ)
reps = 5
reprep = 6

pls = []

theme(:bright)

for k = 1:reprep

    sols = []
    ρ = []
    for i = 1:reps
        push!(sols, solve(prob, EM(), dt = dt))
        push!(ρ, sols[i].u)
    end

    push!(
        pls,
        plot(
            sols[1].t,
            ρ,
            ylim = (-1.05, 1.05),
            background_color = :transparent,
            background_color_inside = :transparent,
            labels = false,
            linewidth = 0.75,
            xticks = 0:5000:10^4,
            linealpha = 0.6,
            xlab = "Time",
            ylab = "Correlation",
        ),
    )

end

# have a look at several options
plot(pls...)

# pick one that clearly communicates the variation of dynamics
pls[4]
