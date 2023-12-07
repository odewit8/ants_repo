using LinearAlgebra
using PyPlot
using LaTeXStrings
using Statistics
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

function sigmaAC(v0,γ,D_T,λ,ω)
    return -0.5*ω^2*v0^2+(γ*v0*ω^2)/(2*(ω^2+1))-ω^2*D_T
end

ω = 2*pi
D_T = 0.01
D_R = 1.0
D = 1.0
η = 1.0
α = 1.0
fast = 1/(2*pi)

howmany = 200
maxgamma = 500
maxv0 = 10
# Fig 3a
# λ = 0.0
# Fig 3b
λ = 0.1
vs = range(0,maxv0,howmany)
γs = range(0,maxgamma,howmany)

heatmaps = []
# Fig 3a
# Ns = [2,8,40]
# Fig 3b
Ns = [2,3,4,7,8,40]

for N ∈ Ns
    heatmap = zeros(howmany,howmany)
    for i ∈ 1:size(vs)[1], j ∈ 1:size(γs)[1]
        γ = γs[j]
        v0 = vs[i]
        M1 = construct_M(N,ω,D_T,D_R,D,α,η,γ,v0,λ)
        bigλ = eigvals(M1)[end]
        heatmap[i,j] = real(bigλ)
    end
    push!(heatmaps,heatmap)
end

heatmapσ0 = zeros(howmany,howmany)
for i ∈ 1:size(vs)[1], j ∈ 1:size(γs)[1]
    γ = γs[j]
    v0 = vs[i]
    heatmapσ0[i,j] = sigmaAC(v0,γ,D_T,λ,ω)
end

fig, axs = plt.subplots(1, 1, figsize=(5,4))
images = []
labels1 = []
# Fig 3a styles
# linestyles1 = [[(0, (1.0, 3.0))],"solid","solid"]
# linewidths1 = [5.0,2.0,2.0]
# colors1 = ["springgreen","red","black"]

# Fig 3b styles
linestyles1 = [[(0, (1.0, 3.0))],"dashdot","dotted","solid","solid","solid"]
linewidths1 = [4.0,1.5,1.5,2.0,1.0,1.5]
colors1 = ["springgreen","deepskyblue","orange","grey","red","black"]

for iN ∈ 1:size(Ns)[1]
    push!(images,axs.contour(heatmaps[iN],levels=[0.0],linewidths=linewidths1[iN],linestyles=linestyles1[iN],colors=colors1[iN],zorder=2*iN))
    h,_ = images[iN].legend_elements()
    push!(labels1,h[1])
end

# Fig 3a
# push!(images,axs.contour(heatmapσ0,levels=[0.0],colors="violet",linestyles=[(0, (0.5, 1.0))],linewidths=4.0,zorder=8))
# h,_ = images[end].legend_elements()
# push!(labels1,h[1])

axs.set_ylim(0,howmany-1)
axs.set_xlim(0,howmany-1)
axs.set_ylabel(L"$\mathrm{Pe}$",fontsize=16)
axs.set_xlabel(L"$\gamma$",fontsize=16)
axs.set_xticks(ticks=range(0,howmany-1,5),labels=range(0,maxgamma,5),fontsize=12)
axs.set_yticks(ticks=range(0,howmany-1,5),labels=range(0,maxv0,5),fontsize=12)
axs.legend(labels1, [L"n=%$(Ns[1])",L"n=%$(Ns[2])",L"n=%$(Ns[3])",L"n=%$(Ns[4])",L"n=%$(Ns[5])",L"n=%$(Ns[6])"],loc="lower right",framealpha=1.0,fontsize=10)
# axs.legend(labels1, [L"n=%$(Ns[1])",L"n=%$(Ns[2])",L"n=%$(Ns[3])",L"\sigma_{\mathrm{AC}}=0"],loc="lower right",framealpha=1.0,fontsize=10)

display(fig)
# fig.savefig("linear_instabs_λ=$(λ)_D_T=$(D_T)_k.eps",bbox_inches="tight")
# fig.savefig("linear_instabs_λ=$(λ)_D_T=$(D_T)_k_AC2.eps",bbox_inches="tight")
close(fig)