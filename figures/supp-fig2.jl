#
# this script generates figure 2 of the supplement
#

include("../solvers/diy/diy_fcts.jl")
using Plots, PlotThemes

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
