using Distributed
@everywhere using DrWatson
@everywhere begin
        quickactivate(@__DIR__, "Ant_Model")
        using Parameters
        include(srcdir("antmodel_pde.jl"))
        include(srcdir("accessory","config.jl"))
end

params = dict_list(Dict(
    :Tf => 5.0,
    :Model => :modelParaEllplusLA,
    :Δt => 1e-5,
    :v0 => [1.5,3.5],
    :DR => 1.0,
    :DT => 0.01,
    :D => 1.0,
    :α => 1.0,
    :η => 1.0,
    :γ => 325.0, 
    :Nx => 31,
    :Ny => 31,
    :Nθ => 21,
    :λ1 => 0.1, 
    :IC => "rand_unif_seed"
    :seed1 => 4472,
    :saveplot => true
)) .|> (p->MyParams(; pairs(p)...))


iout = pmap(run_or_load_active_pde, params);