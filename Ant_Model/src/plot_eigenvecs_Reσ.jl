using LinearAlgebra
using PyPlot
using LaTeXStrings
using Statistics
using Printf
using PyCall
@pyimport numpy as np
rc("text", usetex=false) 
rc("mathtext", fontset="cm")
rc("font", family="serif", serif="cmr10", size=12)
rc("axes.formatter", use_mathtext = true)

function construct_M(N,ω,D_T,D_R,D,α,η,γ,v0,λ)
    M = zeros(N,N)
    du = ones(N-1)*(-im*0.5*ω*v0)
    dl = ones(N-1)*(-im*0.5*ω*v0)
    d = [-ω^2*D_T-(k-1)^2*D_R for k ∈ 1:N]
    d = convert(Vector{ComplexF64},d)
    M = Tridiagonal(dl,d,du)
    M = convert(Matrix{ComplexF64},M)
    M[2,1] += -im*0.5*ω*v0
    M[2,1] += im*2*pi*fast*γ*ω*η/(D*ω^2+α)
    if N >= 3
        M[3,1] += -λ*2*pi*fast*γ*ω^2*η/(D*ω^2+α)
    end
    return M
end

function sigma0(v0,D_T,λ)
    return (4*pi^2+1)*(2*v0^2-4*D_T)/(2*v0+pi^2*λ*v0^2)
end

function coss(θ::Float64,L::Int64)::Vector{Float64}
    return [cos((k-1)*θ) for k ∈ 1:L]
end

m=1
ω = m*2*pi
D_R = 1.0
D = 1.0
η = 1.0
α = 1.0
fast = 1/(2*pi)

howmany = 100
maxgamma = 500
maxv0 = 15
# λs = [0.1]

vs = range(0,maxv0,howmany)
γs = range(0,maxgamma,howmany)

function ploteigf2(N,ω,D_T,D_R,D,α,η,γ,v0,λ)
    M1 = construct_M(N,ω,D_T,D_R,D,α,η,γ,v0,λ)
    v4 = eigvecs(M1)[:,end]
    # println(eigvals(M1))
    σ_n = eigvals(M1)[end]
    f0 = (x,y,θ) -> real(cos(ω*x)*dot(v4,coss(θ,N)))

    howmany2 = 200
    xs2 = range(-0.5,0.5,howmany2)
    θs = range(-0.5*pi,(3/2)*pi,howmany2)
    us21 = [f0(x,0.0,θ) for x ∈ xs2, θ ∈ θs]
    θ2, X = np.meshgrid(θs,xs2)
    usmean = mean(us21,dims=2)

    fig, axs = plt.subplots(1, 1, figsize=(5,4))
    images = []
    # push!(images,axs.imshow(us21,origin="lower",vmin=-1.0,vmax=1.0))
    push!(images,axs.contourf(θ2,X,us21,levels=20,vmin=-1.0,vmax=1.0,extend="both"))
    fig.suptitle(L"$f_n(x,\theta),D_T=%$(D_T),v_0=%$(round(v0,digits=1)),\gamma=%$(round(γ,digits=1)),\lambda=%$(λ),n=%$(N),\Re(\sigma_{n})=%$(round(real(σ_n),digits=2))$",fontsize=10)
    # axs.set_yticks(ticks=range(0,howmany2-1,5),labels=range(-0.5,0.5,5),fontsize=10)
    # axs.set_xticks(ticks=range(0,howmany2-1,5),labels=[L"$0\pi$",L"$\frac{1}{2}\pi$",L"$\pi$",L"$\frac{3}{2}\pi$",L"$2\pi$"],fontsize=10)
    axs.set_yticks(ticks=range(-0.5,0.5,5),labels=range(0.0,1.0,5),fontsize=10)
    axs.set_xticks(ticks=range(-0.5*pi,(3/2)*pi,5),labels=[L"$0\pi$",L"$\frac{1}{2}\pi$",L"$\pi$",L"$\frac{3}{2}\pi$",L"$2\pi$"],fontsize=10)
    axs.set_ylabel(L"$y$")
    axs.set_xlabel(L"$\theta$")
    # axs.axis("square")
    plt.colorbar(images[1],fraction=0.046, pad=0.04,extend="both")
    display(fig)
    fig.savefig("eigf_D_T=$(D_T)_v_0=$(round(v0,digits=1))_γ=$(round(γ,digits=1))_λ=$(λ)_n=$(N).png",bbox_inches="tight")
    fig.suptitle("")
    fig.savefig("eigf_D_T=$(D_T)_v_0=$(round(v0,digits=1))_γ=$(round(γ,digits=1))_λ=$(λ)_n=$(N).eps",bbox_inches="tight")
    close(fig)
    # fig, axs = plt.subplots(1, 1, figsize=(6,4))
    # images = []
    # push!(images,axs.plot(xs2,dropdims(usmean,dims=2)))
    # fig.suptitle(L"$\rho_n(x),D_T=%$(D_T),v_0=%$(round(v0,digits=1)),\gamma=%$(round(γ,digits=1)),\lambda=%$(λ),n=%$(N),\Re(\sigma_{n})=%$(round(real(σ_n),digits=2))$",fontsize=10)
    # axs.set_xticks(ticks=range(0,1,5),labels=range(0.0,1.0,5),fontsize=10)
    # axs.set_xlabel(L"$x$")
    # display(fig)
    # # fig.savefig("eigf_rho_D_T=$(D_T),v_0=$(round(v0,digits=1)),γ=$(round(γ,digits=1)),λ=$(λ),n=$(N).png",bbox_inches="tight")
    # close(fig)
end

# N = 3
γ = γs[80]
γ = 400.0
# v0 = vs[5]
# for λ=0: N=10/3: two lines with attracting polarities to one line with stabilising polarities only
# for λ=0: N=10/3: only a smoothing effect for D_T
# λ=0.1: N=3: same effect but now instead one line with bipolarity
# λ=0.1: N=9: same effect
# there's only one positive eigenvalue of all the eigenvalues for N=9

Ns = [10]
λs = [0.0,0.1]
iv0s = [1]
vs = [8.0]
# D_Ts = [0.001,0.01,0.1,1.0]
D_Ts = [0.01]
for N ∈ Ns
    for λ ∈ λs
        for iv0 ∈ iv0s
            for D_T ∈ D_Ts
                ploteigf2(N,ω,D_T,D_R,D,α,η,γ,vs[iv0],λ)
            end
        end
    end
end