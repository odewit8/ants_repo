using DrWatson
@quickactivate "Ant_Model"

using Parameters

@with_kw struct MyParams
        # volume fraction
        phi::Float64; @assert phi<1&&phi>=0
        # model (2, 3, 4)
        Model::Symbol;
        # initial condtion
        IC::String = "Unif"
        IC_chem::String = "chem0"
        # initial perturbation size
        δ::Float64 = 0.01
        # final time
        Tf::Float64 = 0.01
        # number of time outputs
        NumOut::Int = 50
        # timestep
        Δt::Float64 = 1e-4
        # translational diffuion
        DT::Float64 = 1
        # rotational diffusion
        DR::Float64 = 1
        # c diffusion
        D::Float64 = 1
        # c decay
        α::Float64 = 1
        # c source 
        η::Float64 = 1
        # c interaction
        γ::Float64 = 1
        # self-propulsion
        v0::Float64 = 1; @assert v0>=0
        Nx::Int = 11        # number of x-bins
        Ny::Int = 11         # number of Y-bins
        Nθ::Int = 11         # number of angular-bins
        λ1::Float64 = 0.1
        ϵ::Float64 = 0.0001
        # seed1::Int = 119
        # whether to save figs
        saveplot::Bool = false
        # whether to save data
        savedata::Bool = true
end

struct PDEsim
        F::Array{Float64,4}
        C::Array{Float64,3}
        Rho::Array{Float64,3}
        P::Array{Float64,4}
        T::Array{Float64,1}
        mob_neg::Int
        Δt_f::Float64
end