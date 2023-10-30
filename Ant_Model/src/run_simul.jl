using Distributed
@everywhere using DrWatson
@everywhere begin
        quickactivate(@__DIR__, "Ant_Model")
        using Parameters
        include(srcdir("antmodel_pde.jl"))
        include(srcdir("accessory","config.jl"))
end

params = dict_list(Dict(
    :phi => 0.0, 
    :Tf => 4.0,
    :Model => :modelParaEllplusLA,
    :Δt => 1e-5,
    :v0 => [1.0,6.0],
    :DR => 1.0,
    :DT => 0.01,
    :D => 1.0,
    :α => 1.0,
    :η => 1.0,
    :γ => 300, 
    :Nx => 31,
    :Ny => 31,
    :Nθ => 21,
    :λ1 => 3.1, #0.93,
    :ϵ => 0.0, 
    :IC => ["circle7"],
    :IC_chem => "chem0",
    :δ => 0.0,
    # :seed1 => 1,
    :saveplot => true
)) .|> (p->MyParams(; pairs(p)...))


iout = pmap(run_or_load_active_pde, params);