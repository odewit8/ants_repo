using DrWatson
using LinearAlgebra
using DelimitedFiles
using Interpolations
using JLD2
using ToeplitzMatrices
using Kronecker
@quickactivate "Ant_Model"

include(srcdir("accessory","basic_functions.jl"));
include(srcdir("accessory","PDE_InitialConditions.jl"));
include(srcdir("accessory","config.jl"))


struct params_solver{model_type}
    Tf::Float64
    Δt::Float64
    Δt_out::Float64
    NumOut::Int64
    DT::Float64
    DR::Float64
    v0::Float64
    D::Float64
    α::Float64
    η::Float64
    γ::Float64
    Nx::Int64
    Ny::Int64
    Nθ::Int64
    λ1::Float64
    Cosθ::Array{Float64,1}
    Sinθ::Array{Float64,1}
    Cosθ2::Array{Float64,1}
    Sinθ2::Array{Float64,1}
    Δx::Float64
    Δy::Float64
    Δθ::Float64
end

# produce_or_load version of top function
function run_or_load_active_pde(Params::MyParams)

    @unpack IC = Params

    if IC == "Pert_general" # random initial data
        println("Simulation might already exist but doing a new sample")
        dictout = run_active_pde(Params)
        safesave( datadir("PDE_Sims", savename(Params, "jld2"; sigdigits = 3, ignores = [:NumOut, :saveplot, :savedata])),
                    dictout);
    else
        produce_or_load(run_active_pde,
                    datadir("PDE_Sims"),
                    Params,
                    ignores = [:NumOut, :saveplot, :savedata],
                    sigdigits = 3,
                    tag = false)
    end

end


function run_active_pde(Params::MyParams)

    @unpack Model, IC, Tf, NumOut, Δt, DT, DR, D, α, η, γ, v0, Nx, Ny, Nθ, λ1, savedata = Params

    L::Float64 = 1.0 #length of the interval
    sc::Float64 = 1.0

    xinf  = range(-sc*L/2, stop=sc*L/2, length=Nx+1);
    yinf  = range(-L/2, stop=L/2, length=Ny+1);
    θinf  = range(-pi, stop=pi,  length=Nθ+1);

    x = (xinf[1:end-1] + xinf[2:end]) / 2.0;
    y = (yinf[1:end-1] + yinf[2:end]) / 2.0;
    θ = (θinf[1:end-1] + θinf[2:end]) / 2.0;

    Cosθ = cos.(θ); Sinθ = sin.(θ);
    Cosθ2 = cos.(θinf[1:end-1]); Sinθ2 = sin.(θinf[1:end-1]);

    Δx = step(x);
    Δy = step(y);
    Δθ = step(θ);

    # frequency of output
    Δt_out = Tf/(NumOut-1)
    Δt = min(Δt, Δt_out)

    f0 = ic_PDE(Params, L; IC = IC);

    f = Array{Float64,3}(undef, Nx, Ny, Nθ);
    f = [f0(x̃, ỹ, θ̃) for x̃ ∈ x, ỹ ∈ y, θ̃ ∈ θ];
    c = Array{Float64,2}(undef, Nx, Ny);
    c0 = (x,y) -> 0.0*x;
    c = [c0(x̃, ỹ) for x̃ ∈ x, ỹ ∈ y];
    if sum(f) > 0
        f = f / (Δx * Δy * Δθ *sum(f));
    end

    ParamsSolver = params_solver{Model}(Tf, Δt, Δt_out, NumOut, DT, DR, v0, D, α, η, γ,  Nx, Ny, Nθ,
     λ1, Cosθ, Sinθ, Cosθ2, Sinθ2, Δx, Δy, Δθ)
    println("Running v0=$(v0), γ=$(γ)")
    pdesim::PDEsim = Euler_model(f, c, ParamsSolver)

    @unpack T, F, Δt_f = pdesim;

    println("Done case: $(savename(Params; sigdigits = 3, accesses = [:Model, :v0])). Tf = $(T[end]), Δt_f = $(Δt_f).")

    dictout = @strdict x y θ pdesim L Δx Δy Δθ Nx Ny Nθ
    return dictout
end

function Euler_model(f::Array{Float64,3}, c::Array{Float64,2}, Params::params_solver{:modelParaEll})::PDEsim

    @unpack Tf, Δt, Δt_out, NumOut, DT, DR, D, α, η, γ, v0, Nx, Ny, Nθ,
     λ1, Δx, Δy, Δθ, Cosθ, Sinθ, Cosθ2, Sinθ2 = Params

    t::Float64 = 0.0;

    logtol::Float64 = log(1e-10);

    T = Array{Float64,1}(undef, NumOut)
    F = Array{Float64,4}(undef, NumOut, Nx, Ny, Nθ)
    C = Array{Float64,3}(undef,NumOut,Nx,Ny)
    Rho = Array{Float64,3}(undef, NumOut, Nx, Ny)
    P = Array{Float64,4}(undef, NumOut, 2, Nx, Ny)
    Ux = Array{Float64,3}(undef, Nx, Ny, Nθ);
    Uy = Array{Float64,3}(undef, Nx, Ny, Nθ);
    Uθ = Array{Float64,3}(undef, Nx, Ny, Nθ);
    logf = Array{Float64,3}(undef, Nx, Ny, Nθ);
    ρ = Array{Float64,3}(undef, Nx, Ny, 1);
    ρ = sum(f, dims = 3) * Δθ;

    #---
    #creating the finite difference matrix for the c equation
    col1 = zeros(Nx)
    col1[1] = -2; col1[2] = 1; col1[end] = 1;
    M_fd_x = (D/(Δx^2))*Toeplitz(col1,col1)
    col2 = zeros(Ny)
    col2[1] = -2; col2[2] = 1; col2[end] = 1;
    M_fd_y = (D/(Δy^2))*Toeplitz(col2,col2)
    M_fd_x = convert(Matrix{Float64},M_fd_x)
    M_fd_y = convert(Matrix{Float64},M_fd_y)
    I_x = convert(Matrix{Float64},Diagonal(ones(Nx)))
    I_y = convert(Matrix{Float64},Diagonal(ones(Ny)))
    M_fd = kronecker(M_fd_x,I_y)+kronecker(I_x,M_fd_y)-α*I
    M_fd = Symmetric(M_fd)
    B2 = inv(M_fd)
    #---

    ctr = 1;
    T[ctr] = t;
    F[ctr,:,:,:] = f;
    
    Rho[ctr,:,:] = dropdims(ρ, dims=3);
    c = reshape(-η*B2*vec(Rho[ctr,:,:]),Nx,Ny);
    C[ctr,:,:] = c;
    P[ctr,:,:,:] = polarisation(f, Cosθ, Sinθ, Δθ = Δθ)
    

    while t<Tf
        logf = map(x -> (x>0 ? log(x) : logtol), f);

        Ux = -diffP_left(DT*logf;dims=1, dx = Δx) + v0*(1 .- 0.0.*meanP_left(ρ; dims = 1)).*reshape(Cosθ,1,1,Nθ);
        Uy = -diffP_left(DT*logf;dims=2, dx = Δy) + v0*(1 .- 0.0.*meanP_left(ρ; dims = 2)).*reshape(Sinθ,1,1,Nθ);
        Uθ = -diffP_left(DR*logf;dims=3, dx = Δθ) + γ*(- reshape(Sinθ2,1,1,Nθ).*centdiff(c; dims = 1, dx=Δx) 
             + reshape(Cosθ2,1,1,Nθ).*centdiff(c; dims = 2, dx=Δy));


        a = maximum(abs.(Ux));
        b = maximum(abs.(Uy));
        c̃ = maximum(abs.(Uθ));

        tempu = 6*max(a/Δx, b/Δy, c̃/Δθ);
        tempu2 = max(6*a/Δx, 6*b/Δy, 6*c̃/Δθ, 12*D/((Δx)^2+(Δy)^2))
        # tempu = 4*a/Δx;
        if (1-tempu2*Δt) < 0
            Δt = 1.0/tempu2;
        end

        f = f - Δt*(diffP_right(upwindP!(Ux, f; dims = 1); dims = 1, dx = Δx)
                + diffP_right(upwindP!(Uy, f; dims = 2); dims = 2, dx = Δy)
                + diffP_right(upwindP!(Uθ, f; dims = 3); dims = 3, dx = Δθ)
                );
        
        ρ = sum(f, dims = 3) * Δθ;
        ρ̃ = dropdims(ρ,dims=3);
        c = convert(Array{Float64},reshape(-η*B2*vec(ρ̃),Nx,Ny));
        t  = t + Δt;

        if t > ctr*Δt_out
            ctr = ctr + 1;
            if ctr % 2 == 0
                println(t)
            end
            T[ctr] = t;
            F[ctr,:,:,:] = f;
            C[ctr,:,:] = c;
            Rho[ctr,:,:] = dropdims(ρ, dims=3);
            P[ctr,:,:,:] = polarisation(f, Cosθ, Sinθ, Δθ = Δθ)
        end

    end

    T = T[1:ctr]
    F = F[1:ctr,:,:,:]
    C = C[1:ctr,:,:]
    Rho = Rho[1:ctr,:,:]
    P = P[1:ctr,:,:,:]

    pdesim = PDEsim(F, C, Rho, P, T, Δt)

    return pdesim

end



function Euler_model(f::Array{Float64,3}, c::Array{Float64,2}, Params::params_solver{:modelParaEllplusLA})::PDEsim

    @unpack Tf, Δt, Δt_out, NumOut, DT, DR, D, α, η, γ, v0, Nx, Ny, Nθ,
     λ1, Δx, Δy, Δθ, Cosθ, Sinθ, Cosθ2, Sinθ2 = Params

    t::Float64 = 0.0;

    logtol::Float64 = log(1e-10);

    T = Array{Float64,1}(undef, NumOut)
    F = Array{Float64,4}(undef, NumOut, Nx, Ny, Nθ)
    C = Array{Float64,3}(undef,NumOut,Nx,Ny)
    Rho = Array{Float64,3}(undef, NumOut, Nx, Ny)
    P = Array{Float64,4}(undef, NumOut, 2, Nx, Ny)
    Ux = Array{Float64,3}(undef, Nx, Ny, Nθ);
    Uy = Array{Float64,3}(undef, Nx, Ny, Nθ);
    Uθ = Array{Float64,3}(undef, Nx, Ny, Nθ);
    logf = Array{Float64,3}(undef, Nx, Ny, Nθ);
    ρ = Array{Float64,3}(undef, Nx, Ny, 1);
    ρ = sum(f, dims = 3) * Δθ;
    
    sc = 1.0;
    L = 1.0;
    
    xinf  = range(-sc*L/2, stop=sc*L/2, length=Nx+1);
    yinf  = range(-L/2, stop=L/2, length=Ny+1);
    θinf  = range(-pi, stop=pi,  length=Nθ+1);
    x = (xinf[1:end-1] + xinf[2:end]) / 2.0;
    y = (yinf[1:end-1] + yinf[2:end]) / 2.0;
    θ = (θinf[1:end-1] + θinf[2:end]) / 2.0;

    #---
    #creating the finite difference matrix for the c equation
    col1 = zeros(Nx)
    col1[1] = -2; col1[2] = 1; col1[end] = 1;
    M_fd_x = (D/(Δx^2))*Toeplitz(col1,col1)
    col2 = zeros(Ny)
    col2[1] = -2; col2[2] = 1; col2[end] = 1;
    M_fd_y = (D/(Δy^2))*Toeplitz(col2,col2)
    M_fd_x = convert(Matrix{Float64},M_fd_x)
    M_fd_y = convert(Matrix{Float64},M_fd_y)
    I_x = convert(Matrix{Float64},Diagonal(ones(Nx)))
    I_y = convert(Matrix{Float64},Diagonal(ones(Ny)))
    M_fd = kronecker(M_fd_x,I_y)+kronecker(I_x,M_fd_y)-α*I
    M_fd = Symmetric(M_fd)
    B2 = inv(M_fd)
    #---

    ctr = 1;
    T[ctr] = t;
    F[ctr,:,:,:] = f;
    
    Rho[ctr,:,:] = dropdims(ρ, dims=3);
    c = reshape(-η*B2*vec(Rho[ctr,:,:]),Nx,Ny);
    C[ctr,:,:] = c;
    P[ctr,:,:,:] = polarisation(f, Cosθ, Sinθ, Δθ = Δθ)

    while t<Tf
        logf = map(x -> (x>0 ? log(x) : logtol), f);

        Ux = -diffP_left(DT*logf;dims=1, dx = Δx) + v0*(1 .- 0.0.*meanP_left(ρ; dims = 1)).*reshape(Cosθ,1,1,Nθ);
        Uy = -diffP_left(DT*logf;dims=2, dx = Δy) + v0*(1 .- 0.0.*meanP_left(ρ; dims = 2)).*reshape(Sinθ,1,1,Nθ);
        Uθ = -diffP_left(DR*logf;dims=3, dx = Δθ);
        c3 = interpolatec(c;x,y,θ,Nx,Ny,Nθ,λ1,Δx,Δy);
        Uθ = Uθ + (γ/λ1)*diffP_left(c3;dims=3, dx = Δθ);

        a = maximum(abs.(Ux));
        b = maximum(abs.(Uy));
        c̃ = maximum(abs.(Uθ));
        tempu = 6*max(a/Δx, b/Δy, c̃/Δθ);
        if (1-tempu*Δt) < 0
            Δt = 1.0/tempu2;
        end

        f = f - Δt*(diffP_right(upwindP!(Ux, f; dims = 1); dims = 1, dx = Δx)
                + diffP_right(upwindP!(Uy, f; dims = 2); dims = 2, dx = Δy)
                + diffP_right(upwindP!(Uθ, f; dims = 3); dims = 3, dx = Δθ)
                );
        
        ρ = sum(f, dims = 3) * Δθ;
        ρ̃ = dropdims(ρ,dims=3);
        c = convert(Array{Float64},reshape(-η*B2*vec(ρ̃),Nx,Ny));
        t  = t + Δt;

        if t > ctr*Δt_out
            ctr = ctr + 1;
            if ctr % 10 == 0
                println(t)
            end
            T[ctr] = t;
            F[ctr,:,:,:] = f;
            C[ctr,:,:] = c;
            Rho[ctr,:,:] = dropdims(ρ, dims=3);
            P[ctr,:,:,:] = polarisation(f, Cosθ, Sinθ, Δθ = Δθ)
        end

    end

    T = T[1:ctr]
    F = F[1:ctr,:,:,:]
    C = C[1:ctr,:,:]
    Rho = Rho[1:ctr,:,:]
    P = P[1:ctr,:,:,:]

    pdesim = PDEsim(F, C, Rho, P, T, Δt)

    return pdesim

end