#
# this script generates figure 1 of the supplement
#

include("../solvers/diy/diy_fcts.jl")
using Plots, PlotThemes

p = PARS(n₀ = 10, 𝛍 = zeros(2, 2), T = 100, 𝐆₀ = [1.0 0.5; 0.5 1.0], N = 5000, R = 500);

# initial genetic correlations
ρ₀ˢ = [-0.5 0.0 0.5]

pls = []

for ρ₀ in ρ₀ˢ

    𝐆₀ = [1.0 ρ₀; ρ₀ 1.0]
    p.𝐆₀ = 𝐆₀

    # 𝐆replicating ... 
    𝐆 = 𝐆replicate(p)

    times = p.T * (0:p.N) / p.N
    ρ = 𝐆[:, 2, :] ./ sqrt.(𝐆[:, 1, :] .* 𝐆[:, 3, :])

    pl = plot(
        times,
        ρ,
        ylim = (-1, 1),
        # background_color = :transparent,
        # background_color_inside = :transparent,
        labels = false,
        linewidth = 1.0,
        linecolor = colorant"#3e8fb0",
        linealpha = 0.02,
        xlab = "Time",
        ylab = "Correlation",
    )

    push!(pls, pl)
end

plot(pls..., layout = (3, 1))
