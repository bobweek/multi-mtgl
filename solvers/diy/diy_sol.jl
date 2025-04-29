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

using Plots, PlotThemes

# figure 1

p = PARS(n₀ = 10, 𝛍 = zeros(2, 2), T = 100, 𝐆₀ = [1.0 0.5; 0.5 1.0], N = 5000, R = 500);

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


#
# figure 2
#

p = PARS(n₀ = 10, 𝛍 = zeros(2, 2), T = 100, 𝐆₀ = [1.0 0.5; 0.5 1.0], N = 5000, R = 500);

# 𝐆replicating ... 
𝐆 = 𝐆replicate(p)

# approximate expected dynamics
𝐆m = mean(𝐆, dims = 3)[:, :, 1]

# classical scaling result
Δt = p.T / p.N
δ = p.v / p.n₀
times = 0:Δt:p.T
Pt = p.𝐆₀[1, 1] .* exp.(-δ .* (0:Δt:p.T)) # diag
Qt = p.𝐆₀[1, 2] .* exp.(-δ .* (0:Δt:p.T)) # off-diag

theme(:bright)

plot(
    times,
    𝐆m[:, 1],
    label = "⟨G₁₁⟩",
    xlab = "Time",
    ylab = "Co/Variance",
    background_color = :transparent,
    background_color_inside = :transparent,
)
plot!(times, 𝐆m[:, 2], label = "⟨Ḡ₁₂⟩")
plot!(times, 𝐆m[:, 3], label = "⟨G₂₂⟩")
plot!(times, Pt, label = "𝔼G₁₁=𝔼G₂₂")
plot!(times, Qt, label = "𝔼G₁₂")
