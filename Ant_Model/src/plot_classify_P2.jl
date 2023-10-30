using Distributed

@everywhere include("/home/od275/antmodel/Ant_Model/src/antmodel_pde.jl")
@everywhere include("/home/od275/antmodel/Ant_Model/src/accessory/config.jl")

using PyPlot
using Statistics
using PyCall
using LinearAlgebra
using PyCall
@pyimport matplotlib.colors as colors2
@pyimport numpy as np
@pyimport matplotlib.lines as mlines

rc("text", usetex=false) 
rc("mathtext", fontset="cm")
rc("font", family="serif", serif="cmr10", size=12)
rc("axes.formatter", use_mathtext = true)

# params = dict_list(Dict(
#     :phi => 0.0, 
#     :Tf => 6.0,
#     :Model => :modelParaEll,
#     :Δt => 1e-5,
#     :v0 => [0.1:1.5:10.6...],
#     :DR => 1.0,
#     :DT => 0.0001,
#     :D => 1.0,
#     :α => 1.0,
#     :η => 1.0,
#     :γ => [10.0:50.0:510.0...], 
#     :Nx => 31,
#     :Ny => 31,
#     :Nθ => 21,
#     :λ1 => 0.0, #0.93,
#     :ϵ => 0.0, 
#     :IC => "rand_unif",
#     :IC_chem => "chem0",
#     :δ => 0.0,
#     # :seed1 => 1,
#     :saveplot => true
# )) .|> (p->MyParams(; pairs(p)...))

# params = dict_list(Dict(
#     :phi => 0.0, 
#     :Tf => 5.0,
#     :Model => :modelParaEllplusLA,
#     :Δt => 1e-5,
#     :v0 => [0.5:0.5:10.0...],
#     :DR => 1.0,
#     :DT => 0.01,
#     :D => 1.0,
#     :α => 1.0,
#     :η => 1.0,
#     :γ => [25.0:25.0:500.0...], 
#     :Nx => 31,
#     :Ny => 31,
#     :Nθ => 21,
#     :λ1 => 3.1, #0.93,
#     :ϵ => 0.0, 
#     :IC => "rand_unif",
#     :IC_chem => "chem0",
#     :δ => 0.0,
#     # :seed1 => 1,
#     :saveplot => true
# )) .|> (p->MyParams(; pairs(p)...))

params = dict_list(Dict(
    :phi => 0.0, 
    :Tf => 5.0,
    :Model => :modelParaEll,
    :Δt => 1e-5,
    :v0 => [0.5:1.0:10.0...],
    :DR => 1.0,
    :DT => 0.01,
    :D => 1.0,
    :α => 1.0,
    :η => 1.0,
    :γ => [25.0:50.0:500.0...], 
    :Nx => 31,
    :Ny => 31,
    :Nθ => 21,
    :λ1 => 0.0, #0.93,
    :ϵ => 0.0, 
    :IC => "rand_unif",
    :IC_chem => "chem0",
    :δ => 0.0,
    # :seed1 => 1,
    :saveplot => true
)) .|> (p->MyParams(; pairs(p)...))

iout = pmap(run_or_load_active_pde, params);

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

m=1
ω = m*2*pi
D_T = 0.01
D_R = 1.0
D = 1.0
η = 1.0
α = 1.0
fast = 1/(2*pi)
N=8

@unpack Nx, Ny, IC, Tf,λ1 = params[1]
@unpack x, y, θ, pdesim, Δθ, Δx = iout[1][1];

Cos2θ = cos.(2*θ); Sin2θ = sin.(2*θ);

function polarisation2(f::Array{T,3}, Cos2θ::Array{T,1}, Sin2θ::Array{T,1}; Δθ::Float64)::Array{T,3} where T<:Real
    sizes = size(f);
    p2 = Array{T,3}(undef, 2, sizes[1], sizes[2])

    for i ∈ 1:sizes[1]
        for j ∈ 1:sizes[2]
            p2[1,i,j] = sum(f[i,j,:].*Cos2θ) * Δθ
            p2[2,i,j] = sum(f[i,j,:].*Sin2θ) * Δθ
        end
    end

    return p2
end

howmany = 200
maxgamma = 505
maxv0 = 10.2
λ = 0.0
vs = range(0,maxv0,length=howmany)
γs = range(0,maxgamma,length=howmany)

heatmap = zeros(howmany,howmany)
for i ∈ 1:size(vs)[1], j ∈ 1:size(γs)[1]
    γ = γs[j]
    v0 = vs[i]
    M1 = construct_M(N,ω,D_T,D_R,D,α,η,γ,v0,λ)
    bigλ = eigvals(M1)[end]
    heatmap[i,j] = real(bigλ)
end

ix = 10
iy = 10 
Γ = zeros(ix,iy)
V0s = zeros(ix,iy)
L2s = zeros(ix,iy)
P2s = zeros(ix,iy)

tend = 50
for i ∈ 1:ix
    for j ∈ 1:iy
        k = iy*(i-1)+j
        @unpack γ, λ1, v0 = params[k]
        @unpack pdesim, Δx = iout[k][1];
        @unpack F, C, Rho, P, T = pdesim;
        Γ[i,j] = γ
        V0s[i,j] = v0
        L2s[i,j] = sqrt(sum((F[tend,:,:,:].-(1/(2*pi))).^2))*Δx*sqrt(Δθ)
        P2s[i,j] = norm(sum(polarisation2(F[tend,:,:,:], Cos2θ, Sin2θ, Δθ = Δθ),dims=[2,3]))*Δx^2
    end
end

close(fig)
# fig, axs = plt.subplots(1, 1, figsize=(7.5,5))
fig, axs = plt.subplots(1, 1, figsize=(6,4))
images = []
labels1 =[]
homogs = []
stripes = []
spots = []
unclassified = []

norm1 = plt.Normalize(0,0.92)

# for i ∈ 1:ix
#     for j ∈ 1:ix
#         if L2s[i,j] < 0.001
#             push!(images,axs.scatter(x=Γ[i,j],y=V0s[i,j],s=25,c="grey",marker="s",edgecolors="none"))
#             push!(homogs,ix*(j-1)+i)
#         elseif P2s[i,j] > 0.58
#             push!(images,axs.scatter(x=Γ[i,j],y=V0s[i,j],s=40,c="deepskyblue",marker="^",edgecolors="none"))
#             push!(stripes,ix*(j-1)+i)
#         elseif L2s[i,j] < 0.05
#             push!(images,axs.scatter(x=Γ[i,j],y=V0s[i,j],s=40,c="red",marker="o",edgecolors="none"))
#             # push!(spots,ix*(j-1)+i)
#         else
#             push!(images,axs.scatter(x=Γ[i,j],y=V0s[i,j],s=40,c="violet",marker="o",edgecolors="none"))
#             push!(spots,ix*(j-1)+i)
#         end
#     end
# end

cmap1 = "PiYG"
for i ∈ 1:ix
    for j ∈ 1:iy
        if L2s[i,j] < 0.001
            push!(images,axs.scatter(x=[Γ[i,j]],y=[V0s[i,j]],s=40,c=[P2s[i,j]],cmap=cmap1,norm=norm1,marker="s",edgecolors="none"))
            push!(homogs,ix*(j-1)+i)
        elseif P2s[i,j] > 0.58
            push!(images,axs.scatter(x=[Γ[i,j]],y=[V0s[i,j]],s=40,c=[P2s[i,j]],cmap=cmap1,norm=norm1,marker="^",edgecolors="none"))            
            push!(stripes,ix*(j-1)+i)
        elseif L2s[i,j] < 0.05
            push!(images,axs.scatter(x=[Γ[i,j]],y=[V0s[i,j]],s=40,c=[P2s[i,j]],cmap=cmap1,norm=norm1,marker="x",edgecolors="none"))
            push!(unclassified,ix*(j-1)+i)
        else
            push!(images,axs.scatter(x=[Γ[i,j]],y=[V0s[i,j]],s=40,c=[P2s[i,j]],cmap=cmap1,norm=norm1,marker="o",edgecolors="none"))            
            push!(spots,ix*(j-1)+i)
        end
    end
end

y1 = range(0,maxv0,length=howmany)
x1 = range(0,maxgamma,length=howmany)
X, Y = np.meshgrid(x1,y1)
push!(images,axs.contour(X,Y,heatmap,levels=[0.0],linewidths=1.5,linestyles="solid",colors="black"))
# h,_ = images[end].legend_elements()
# push!(labels1,h[1])
# push!(images,axs.scatter(x=Γ[:],y=V0s[:],c=L2s[:],cmap="viridis"))
# push!(images,axs.scatter(x=Γ[:],y=V0s[:],c=P2s[:],cmap="viridis"))
axs.set_xlabel(L"\gamma")
axs.set_ylabel(L"\mathrm{Pe}")
axs.set_xlim(0,maxgamma)
axs.set_ylim(0,maxv0)
# grey_squares = mlines.Line2D([], [], color="grey", marker="s", linestyle="None", markersize=5, label=L"f_\ast")
# orange_circles = mlines.Line2D([], [], color="violet", marker="o", linestyle="None", markersize=5, label=L"\mathrm{spot}")
# blue_triangles = mlines.Line2D([], [], color="deepskyblue", marker="^", linestyle="None", markersize=5, label=L"\mathrm{stripe}")
# red_cross = mlines.Line2D([], [], color="red", marker="x", linestyle="None", markersize=5, label=L"\mathrm{stripe}")
# black_line = mlines.Line2D([], [], color="black", marker="_", linestyle="None", markersize=15, label=L"\Re(\sigma_{n=8})=0")
grey_squares = mlines.Line2D([], [], marker="s", mec="black", mfc="white",linestyle="None", markersize=5, label=L"f_\ast")
orange_circles = mlines.Line2D([], [], mec="black", mfc="white", marker="o", linestyle="None", markersize=5, label=L"\mathrm{spot}")
blue_triangles = mlines.Line2D([], [], mec="black", mfc="white", marker="^", linestyle="None", markersize=5, label=L"\mathrm{stripe}")
red_cross = mlines.Line2D([], [], mec="black", mfc="white", marker="x", linestyle="None", markersize=5, label=L"\mathrm{unclassfied}")
black_line = mlines.Line2D([], [], mec="black", mfc="white", marker="_", linestyle="None", markersize=15, label=L"\mathrm{linear}")
# black_line = mlines.Line2D([], [], mec="black", mfc="white", marker="_", linestyle="None", markersize=15, label=L"\mathrm{Re}(\sigma_{n=8})=0")
axs.legend(handles=[grey_squares, orange_circles, blue_triangles,red_cross,black_line],framealpha=1.0,loc="upper left",fontsize=8)
cb = fig.colorbar(images[5],ax=axs,shrink=0.8)
fig.axes[2].set(title=L"|\mathbf{P}_2|")
# cb.set_xlabel(L"|\mathbf{P}_2|")
# fig.suptitle(L"\|\rho(t=5.0)-1\|_{L^2},D_T=10^{-2},\alpha=1.0,\lambda=0.1,\Delta t=10^{-5},N_x=N_y=31,N_\theta=21",fontsize=10)
display(fig)
fig.savefig("paper_images/classify_Pe=10_gamma=500_Tf=5_DT=0.01_lambda=0.png")
fig.savefig("paper_images/classify_Pe=10_gamma=500_Tf=5_DT=0.01_lambda=0.eps")
# fig.savefig("paper_images/classify_Pe=10_gamma=500_Tf=5_DT=0.0001_lambda=0.png")
# fig.savefig("paper_images/classify_Pe=10_gamma=500_Tf=5_DT=0.0001_lambda=0.eps")

