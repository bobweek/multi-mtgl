#
# making use of "CANNED" methods in julia
#     to numerically solve for genetic correlations
#

using StochasticDiffEq, Plots, Statistics, Printf

# theme(:rose_pine_dawn)
theme(:bright)
# theme(:default)

function f(ρ, δ, t)
    -δ * ρ * (1 - ρ^2) / 2
end

function g(ρ, δ, t)
    √δ * (1 - ρ^2)
end

# Custom engineering notation formatter
# function eng_format(x)
#     return @sprintf("%.2g", x)  # Uses 3 significant figures
# end

# Custom engineering notation formatter (no '+' sign, single-digit exponent)
function eng_format(x)
    if x == 0
        return "0"
    end
    exp = floor(Int, log10(abs(x)) ÷ 3 * 3)  # Get exponent in multiples of 3
    base = x / 10^exp  # Normalize base to engineering notation

    if exp == 0
        return @sprintf("%.3g", base)  # No exponent if it's 10^0
    else
        return @sprintf("%.3ge%d", base, exp)  # Ensures a clean single-digit exponent
    end
end

#
# figure 1
#

ρ₀ = 0.0
δ = 0.001
dt = 0.1
tspan = (0.0, 10.0^4)

prob = SDEProblem(f, g, ρ₀, tspan, δ)
reps = 5
reprep = 6

pls = []

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
            linewidth = 0.4,
            xticks = 0:5000:10^4,
            xformatter = eng_format,
            # linecolor=colorant"#3e8fb0",
            linealpha = 0.6,
            xlab = "Time",
            ylab = "Correlation",
        ),
    )

end

plot(pls...)

#
# figure 2
#

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
        xformatter = eng_format,
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


#
# figure 3
#

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
    ylabel = "Proportion spent near ±1",
)
