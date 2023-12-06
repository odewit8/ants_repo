# Initial data
using Distributions
using Random

include(srcdir("accessory","config.jl"))

function ic_PDE(Params::MyParams,L::Float64; IC = "Unif")
    if IC == "Unif"
        P = (x,y,θ) -> 0.0*x + 1.0;
    elseif IC == "Unifzero"
        P = (x,y,θ) -> 0.0*x;
    elseif IC == "rand_unif"
        P = (x,y,θ) -> rand(Float64);
    elseif IC == "rand_unif_seed"
        # if seed1 is used (change config and Parameters by uncommenting #seed1):
        @unpack seed1 = Params
        Random.seed!(seed1)
        P = (x,y,θ) -> rand(Float64);
    else
        println("Unknown initial condition")
    end
end