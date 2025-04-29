#
# this script produces the right panel of figure 1 in the main text
#

using DifferentialEquations, ModelingToolkit, Plots, Distributions
using MethodOfLines: MOLFiniteDifference

#
# define the forwards kolmogorov pde
#

@parameters t ρ
@variables p(..)
Dt = Differential(t)
Dρ = Differential(ρ)
Dρ2 = Differential(ρ)^2

ξ = 1.0  # = v/n

μ(ρ) = -(1/2) * ξ * ρ * (1 - ρ^2)
D(ρ) = sqrt(ξ) * (1 - ρ^2)

# the pde
drift_term = -Dρ(μ(ρ) * p(t, ρ))
diffusion_term = (1/2) * Dρ2((D(ρ))^2 * p(t, ρ))

eq = Dt(p(t,ρ)) ~ drift_term + diffusion_term

#
# boundary and initial conditions
#

function initial_condition(ρ)
    μ = 0.5
    σ = 0.001
    gauss = pdf(Normal(μ, σ), ρ)
    cutoff = cdf(Normal(μ, σ), 1) - cdf(Normal(μ, σ), -1)
    return gauss / cutoff
end

bcs = [p(0, ρ) ~ initial_condition(ρ)]
domains = [t ∈ IntervalDomain(0.0, 2.0), ρ ∈ IntervalDomain(-1.0, 1.0)]
pdesys = PDESystem([eq], bcs, domains, [t, ρ], [p(t, ρ)]; name = :fokker_planck)

#
# discretize and solve
#

dρ = 0.01
discretization = MOLFiniteDifference([ρ => dρ], t)
prob = discretize(pdesys, discretization)
sol = solve(prob, Tsit5(), saveat=0.01)

#
# plot solution
#

pkey = only(filter(k -> startswith(string(k), "p(t"), keys(sol.u)))
p_matrix = sol.u[pkey]
# ρ_vals = collect(-1.0:dρ:1.0)

# number of curves to show
n_curves = 20

# rosé pine-inspired colors
rosepine_purple = colorant"#6e6a86"  # or try "#c4a7e7" for a more rosy tone
rosepine_pine = colorant"#9ccfd8"
rosepine_rose = colorant"#c4a7e7"

color_gradient = cgrad([rosepine_purple, rosepine_pine], n_curves)

# indices of curves to keep (evenly spaced across time)
curve_indices = round.(Int, range(1, size(p_matrix, 1), length=n_curves))

# spatial values
ρ_vals = range(-1.0, 1.0, length=size(p_matrix, 2))

# plot setup
theme(:bright)
pl = plot(legend=false, size=(400, 400), dpi=300, background_color = :transparent)

# plot selected curves
for (j, i) in enumerate(reverse(curve_indices))
    y = p_matrix[i, :]
    x = ρ_vals
    color = color_gradient[j]
    pl = plot!(x, y, color=color, lw=1.2, ylims=(0,10), alpha=0.75)
end

pl = ylabel!("Density of ρ")
pl = xlabel!("Genetic Correlations ρ")

# make legend
x_legend = -0.4
y_start = 6.0
for j in 1:n_curves
    y_pos = y_start + offset * (j - 1)
    color = color_gradient[j]
    plot!(pl, [x_legend, x_legend + 0.05], [y_pos, y_pos], color=color, lw=4)
end

# legend labels
annotate!(pl, x_legend + 0.05, y_start - 0.4,                     Plots.text("t = 2000", :black, 8, :center))
annotate!(pl, x_legend + 0.05, y_start + offset*(n_curves-1) + 0.4, Plots.text("t = 0", :black, 8, :center))

@show pl

savefig(pl, "pρc.png")

run(`notify-send "Julia says:" "Your computation is done!"`)
