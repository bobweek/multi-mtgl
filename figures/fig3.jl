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

dt = 0.5
reps = 3000
𝛒₀ = [-0.9, -0.2, 0.4, 0.8]

pls = []

for ρ₀ in 𝛒₀

    prob = SDEProblem(f, g, ρ₀, tspan, δ)

    sols = []
    ρ = []
    for i = 1:reps
        push!(sols, solve(prob, EM(), dt = dt))
        push!(ρ, sols[i].u)
    end

    pl = plot(
        sols[1].t,
        ρ[1:20],
        ylim = (-1.05, 1.05),
        background_color = :transparent,
        background_color_inside = :transparent,
        labels = false,
        linewidth = 0.4,
        xticks = 0:5000:10^4,
        linecolor = colorant"#3e8fb0",
        linealpha = 0.2,
        xlab = "Time",
        ylab = "Correlation",
    )
    plot!(
        sols[1].t,
        mean(hcat(ρ...), dims = 2),
        linecolor = colorant"#eb6f92",
        labels = false,
    )

    push!(pls, pl)

end

plot(pls...)
