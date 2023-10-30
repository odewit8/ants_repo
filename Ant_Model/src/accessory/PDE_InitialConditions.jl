# Initial data
using Distributions
using Random

include(srcdir("accessory","config.jl"))

function ic_PDE(Params::MyParams, L::Float64; IC = "Unif", var_x = 0.05, var_θ = 0.2, λ = 10)
    sf::Float64 = 2*pi/L; # scaling factor
    # Random.seed!(119)

    if IC == "Unif"
        P = (x,y,θ) -> 0.0*x + 1.0;
    elseif IC == "Unifzero"
        P = (x,y,θ) -> 0.0*x;
    elseif IC == "GaussXTH"
        P = (x,y,θ) -> VMshift(x, sf=sf, μ=0.0, κ=1/var_x)*VMshift(θ, μ=pi/2, κ=1/var_θ);
    elseif IC == "GaussXYTH"
        P = (x,y,θ) -> VMshift(x, sf=sf, μ=0.0, κ=1/var_x)*VMshift(y, sf=sf, μ=0.0, κ=1/var_x)*VMshift(θ, μ=pi/2, κ=1/var_θ);
    elseif IC == "GaussXY"
        P = (x,y,θ) -> VMshift(x, sf=sf, μ=0.0, κ=1/var_x)*VMshift(y, sf=sf, μ=0.0, κ=1/var_x);
    elseif IC == "GaussX"
        P = (x,y,θ) -> VMshift(x, sf=sf, μ=0.0, κ=1/var_x);
    elseif IC == "SquareX"
        P = (x,y,θ) -> abs(x-0.4)<0.1;
    elseif IC == "ic0"
        P = (x,y,θ) -> -L/6 < x < L/6 ? (1-abs(6*x/L))*VMshift(θ, μ=pi/2, κ=1/var_θ) : 0;
    elseif IC == "circle4"
        P = (x,y,θ) -> x^2 + y^2 < (L/4)^2 ? 1 : 0;
    elseif IC == "circle7"
        P = (x,y,θ) -> x^2 + y^2 < (L/7)^2 ? 1 : 0;
    elseif IC == "circlerand"
        P = (x,y,θ) -> x^2 + y^2 < (L/4)^2 ? rand(Float64) : 0;
    elseif IC == "circle5"
        P = (x,y,θ) -> box(x,y,L)
    elseif IC == "circle_w_p"
        P = (x,y,θ) -> x^2 + y^2 < (L/6)^2 ? VMshift(θ, μ=pi/2, κ=1/var_θ) : 0;
    elseif IC == "rand_unif"
        P = (x,y,θ) -> rand(Float64)
    elseif IC == "rand_unif_seed"
        # if seed1 is used (change config and Parameters by uncommenting #seed1):
        @unpack seed1 = Params
        Random.seed!(seed1)
        P = (x,y,θ) -> rand(Float64)
    elseif IC == "rand_unif_lane"
        @unpack δ = Params
        P = (x,y,θ) -> -L/5 < x < L/5 ? δ + rand(Float64) : rand(Float64);
    elseif IC == "rand_unif_lane_polar"
        P = (x,y,θ) -> -L/5 < x < L/5 ? 5*(0.2 + rand(Float64))*VMshift(θ, μ=pi/2, κ=1/var_θ) : 0.5*rand(Float64);
    elseif IC == "ic0_pert"
        @unpack δ = Params
        P = (x,y,θ) -> -L/6 < x < L/6 ? (1-abs(6*x/L))*VMshift(θ, μ=pi/2, κ=1/var_θ) : δ + δ*(cos(5*sf*x)*cos(5*sf*y));
    elseif IC == "x_ind7"
        P = (x,y,θ) -> -4L/40 < x <= 4L/40 ? 1 : 0;
    elseif IC == "y_ind"
        P = (x,y,θ) -> -L/6 < y <= L/6 ? 1 : 0;
    elseif IC == "y_ind2"
        P = (x,y,θ) -> -L/6 < y <= L/6 ? 1 : 0;
    elseif IC == "x_ind_rand"
        P = (x,y,θ) -> -L/6 < x <= L/6 ? rand(Float64) : 0;
    elseif IC == "x_ind"
        P = (x,y,θ) -> -L/6 < x <= L/6 ? 1 : 0;
    elseif IC == "y_ind_rand"
        P = (x,y,θ) -> -L/6 < y <= L/6 ? rand(Float64) : 0;
    elseif IC == "Pert3"
        @unpack δ = Params
        P = (x,y,θ) -> (1+δ*(cos(2*pi*x)));
    elseif IC == "Pertk2"
        @unpack δ = Params
        P = (x,y,θ) -> (1+δ*(cos(4*pi*x)));
    elseif IC == "Pertk10"
        @unpack δ = Params
        P = (x,y,θ) -> (1+δ*(cos(20*pi*x)));
    elseif IC == "Pert3TH"
        @unpack δ = Params
        P = (x,y,θ) -> (1+δ*(cos(2*pi*x)))*VMshift(θ, μ=pi/2, κ=1/var_θ);
    elseif IC == "GaussTH"
        P = (x,y,θ) -> VMshift(θ, μ=pi/2, κ=1/var_θ);
    elseif IC == "sinX"
        P = (x,y,θ) -> sin(pi*x/L)^2;
    elseif IC == "test_blob1"
        Random.seed!(12)
        P = (x,y,θ) -> 0 < x < 2*L/5 ? rand(Float64) : 0;
    elseif IC == "test_blob2"
        Random.seed!(12)
        P = (x,y,θ) -> -L/5 < x < L/5 ? rand(Float64) : 0;
    elseif IC == "tilted2"
        P = (x,y,θ) -> 0.0*x + 1.0;
    elseif IC == "tilted"
        P =  (x,y,θ) -> 0.0*x + 1.0;
    elseif IC == "Pert_general"
        @unpack δ = Params
        P = (x,y,θ) -> 1.0 + δ*sin(rand(1:λ)*pi*x)*sin(rand(1:λ)*pi*y)*sin(rand(1:λ)*θ/2);
    else
        println("Unknown initial condition")
    end
end

function VMshift(x::Float64; sf::Float64=1.0, μ = 0.0, κ = 1.0)::Float64
    return pdf(VonMises(μ*sf,κ/sf^2), mod(x*sf-(μ*sf-pi),2*pi) + (μ*sf-pi))*sf
end



function ic_Chem(Params::MyParams,L::Float64; IC_chem="chem4")
    @unpack α, Model = Params
    if IC_chem == "chem3"
        C = (x,y) -> sin(pi*(x+0.5))^2;
    elseif IC_chem == "chem4"
        C = (x,y) -> x + L/2;
    elseif IC_chem == "cos_2"
        C = (x,y) -> cos(pi*x)^2;
    elseif IC_chem == "chem5"
        C = (x,y) -> L/2-abs(x);
    elseif IC_chem == "chem0"
        C = (x,y) -> 0.0*x; #otherwise int type
    elseif IC_chem == "hom"
        C = (x,y) -> 2*pi/α;
    elseif IC_chem == "chem_pert"
        @unpack δ = Params
        C = (x,y) ->  δ*(cos(2*pi*x)+1);
    end
end