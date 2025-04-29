#
# this script solves the model defined in "diy_fcts.jl"
#

include("diy_fcts.jl")
using Plots

p = PARS()

#
# evolve whole system
#

n, 𝐳̄, 𝐆 = evolve(p);

Δt = p.T / p.N;
δ = p.v / p.n₀;
times = 0:Δt:p.T;

pln = plot(times,n,legend=false,xlabel=" ",ylabel="Abundance");
plz = plot(times,𝐳̄,legend=false,xlabel=" ",ylabel="Mean\nTraits");
plG = plot(times,𝐆,legend=false,xlabel="Time",ylabel="Genetic\n(Co)Variances");
plot([pln plz plG]...,layout=(3,1))

#
# run 𝐆replicates (only 𝐆 evolves)
#

𝐆 = drift𝐆(p);
plot(times,𝐆,legend=false,xlabel="Time",ylabel="Genetic (Co)Variances")
