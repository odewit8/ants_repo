using LinearAlgebra
using PyPlot
using LaTeXStrings
using Statistics
using PyCall
@pyimport numpy as np
rc("text", usetex=false) 
rc("mathtext", fontset="cm")
rc("font", family="serif", serif="cmr10", size=12)
rc("axes.formatter", use_mathtext = true)

function construct_Mn(N,ω,D_T,α,γ,Pe,λ)
    M = zeros(N,N)
    a = -ω^2*D_T
    b = -1im*0.5*ω*Pe
    C_1 = γ*ω/(ω^2+α)
    du = ones(N-1)*(b)
    dl = ones(N-1)*(b)
    d = [a-(k-1)^2 for k ∈ 1:N]
    d = convert(Vector{ComplexF64},d)
    M = Tridiagonal(dl,d,du)
    M = convert(Matrix{ComplexF64},M)
    M[2,1] += b
    M[2,1] += 1im*C_1
    if N >= 3
        M[3,1] += -λ*ω*C_1
    end
    return M
end

function coss(θ::Float64,L::Int64)::Vector{Float64}
    return [cos((k-1)*θ) for k ∈ 1:L]
end

ω = m*2*pi
D_R = 1.0
D = 1.0
η = 1.0
α = 1.0
fast = 1/(2*pi)

N = 40
λ = 0.1
Pe = 3.5
D_T = 0.01
γ = 325
M1 = construct_Mn(N,ω,D_T,α,γ,Pe,λ)
v = eigvecs(M1)[:,N]
σ_n = eigvals(M1)[N]
v ./= v[1]

λ = 0.1
M1 = construct_Mn(N,ω,D_T,α,γ,Pe,λ)
v = eigvecs(M1)[:,N]
v ./= v[1]

f0 = (x,θ) -> real(exp(1im*ω*x)*sum(v.*coss(θ,N)))
howmany2 = 1000
xs2 = range(-0.5,0.5,howmany2)
θs = range(-pi,pi,howmany2)
θ2 = [t for x∈xs2, t∈θs];
X = [x for x∈xs2, t∈θs];
z = [f0(x,t) for x∈xs2, t∈θs]

fig, axs = plt.subplots(1, 1, figsize=(5,4))
images = []
push!(images,axs.contourf(X,θ2,z,levels=20))
axs.set_xlabel(L"$x$")
axs.set_ylabel(L"$\theta$")
axs.set_yticks(ticks=range(-pi,pi,5),labels=[L"$-\pi$",L"$-\frac{1}{2}\pi$",L"$0\pi$",L"$\frac{1}{2}\pi$",L"$\pi$"],fontsize=12)
axs.set_xticks(ticks=range(-0.5,0.5,5),labels=range(-0.5,0.5,5),fontsize=12)
cb = plt.colorbar(images[1],shrink=0.8)
cb.ax.set_title(L"\widetilde{f}",fontsize=14)
display(fig)
# fig.savefig("eigf_D_T=$(D_T)_Pe=$(Pe)_γ=$(γ)_λ=$(λ)_n=$(N).eps",bbox_inches="tight")
close(fig)

ρ = (x,y) -> real(exp(1im*ω*x)*v[1]*2*pi)
howmany2 = 100
xs2 = range(-0.5,0.5,howmany2)
ys2 = range(-0.5,0.5,howmany2)
X = [x for x∈xs2, y∈ys2];
Y = [y for x∈xs2, y∈ys2];
PRho = [ρ(x,y) for x∈xs2, y∈ys2]

fig, axs = plt.subplots(1, 1, figsize=(5,4))
images = []
push!(images,axs.contourf(X,Y,PRho,levels=20,cmap="Greys",alpha=0.7))
axs.set_xlabel(L"$x$")
axs.set_ylabel(L"$y$")
axs.set_xticks(ticks=range(-0.5,0.5,5),labels=range(-0.5,0.5,5),fontsize=12)
axs.set_yticks(ticks=range(-0.5,0.5,5),labels=range(-0.5,0.5,5),fontsize=12)
cb = plt.colorbar(images[1],shrink=0.8)
cb.ax.set_title(L"\tilde{\rho}",fontsize=14)
display(fig)
# fig.savefig("eigf_rho_D_T=$(D_T)_Pe=$(Pe)_γ=$(γ)_λ=$(λ)_n=$(N).pdf",bbox_inches="tight")
close(fig)