using LinearAlgebra
using PyPlot
using LaTeXStrings
using Statistics
using Printf
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

function sigma02(v0,γ,D_T,λ,ω)
    return -0.5*ω^2*v0^2+(γ*v0*ω^2)/(2*(ω^2+1))-ω^2*D_T+(ω^4*λ*γ*v0^2)/(16*(ω^2+1))
    # fourth order correction:
    # return -0.5*ω^2*v0^2+(γ*v0*ω^2)/(2*(ω^2+1))-ω^2*D_T-ω^4*D_T^2/D_R+(ω^4*λ*γ*v0^2)/(16*(ω^2+1))
end


m=1
ω = m*2*pi
D_T = 0.0
D_R = 1.0
D = 1.0
η = 1.0
α = 1.0
fast = 1/(2*pi)

howmany = 400
maxgamma = 2000
maxv0 = 100
# λs = [0.1]
λ = 0.1
vs = range(0,maxv0,howmany)
γs = range(0,maxgamma,howmany)
# γ = 10
# v0 = 2
# N=10
# γ = 10
# v0 = 2
# M = construct_M(N,ω,D_T,D_R,D,α,η,γ,v0,λ)

heatmaps = []
# Ns = [2,3,4,7,8]
Ns = [11,12]
for N ∈ Ns
    heatmap = zeros(howmany,howmany)
    for i ∈ 1:size(vs)[1], j ∈ 1:size(γs)[1]
        γ = γs[j]
        v0 = vs[i]
        M1 = construct_M(N,ω,D_T,D_R,D,α,η,γ,v0,λ)
        # M1 = construct_M(N,ω,D_T,D_R,D,α,η,0,v0,λ)
        bigλ = eigvals(M1)[end]
        heatmap[i,j] = real(bigλ)
    end
    push!(heatmaps,heatmap)
end

heatmapσλ = zeros(howmany,howmany)
for i ∈ 1:size(vs)[1], j ∈ 1:size(γs)[1]
    γ = γs[j]
    v0 = vs[i]
    heatmapσλ[i,j] = sigma02(v0,γ,D_T,λ,ω)
end

heatmapσ0 = zeros(howmany,howmany)
for i ∈ 1:size(vs)[1], j ∈ 1:size(γs)[1]
    γ = γs[j]
    v0 = vs[i]
    heatmapσ0[i,j] = sigma02(v0,γ,D_T,0,ω)
end

fig, axs = plt.subplots(1, 1, figsize=(6,4))
images = []
labels1 = []
# linestyles1 = ["dashed","dashdot","dotted","solid","solid"]
# linewidths1 = [1.5,1.5,1.5,1.5,1.5]
# colors1 = ["mediumseagreen","deepskyblue","violet","grey","black"]
linestyles1 = [[(0, (0.7, 1.0))],[(0, (4.0, 5.0))]]
linewidths1 = [2.0,3.0]
colors1 = ["mediumspringgreen","violet"]
# push!(images,axs.imshow(heatmap,origin="lower",cmap="viridis"))
for iN ∈ 1:size(Ns)[1]
    push!(images,axs.contour(heatmaps[iN],levels=[0.0],linewidths=linewidths1[iN],linestyles=linestyles1[iN],colors=colors1[iN]))
    h,_ = images[iN].legend_elements()
    push!(labels1,h[1])
end

push!(images,axs.contour(heatmapσ0,levels=[0.0],colors="black",zorder=-1))
h,_ = images[end].legend_elements()
push!(labels1,h[1])
# push!(images,axs.contour(heatmapσλ,levels=[0.0],colors="darkorange"))
# h,_ = images[end].legend_elements()
# push!(labels1,h[1])

xs1 = range(0,howmany-1)
ys1 = ((howmany-1)/maxv0)*(maxgamma/(howmany-1))*(1+2*pi*λ)*xs1./(4*pi^2+1)
# ys1 = range(0,howmany-1)
# xs1 = zeros(howmany)
# for i in 1:howmany
#     xs1[i] = (howmany-1)*sigma0(vs[i],D_T,0.0)/maxgamma
# end
push!(images,axs.plot(xs1,ys1,marker="o",linestyle="none",markevery=9,color="royalblue"))
h2, = images[end]
push!(labels1,h2)
# ys2 = range(0,howmany-1)
# xs2 = zeros(howmany)
# for i in 1:howmany
#     xs2[i] = (howmany-1)*sigma0(vs[i],D_T,λ)/maxgamma
# end
# push!(images,axs.plot(xs2,ys2,marker="^",linestyle="none",markevery=13,color="darkorange"))
# h2, = images[end]
# push!(labels1,h2)
axs.set_ylim(0,howmany-1)
axs.set_xlim(0,howmany-1)
axs.set_ylabel(L"$\mathrm{Pe}$",fontsize=14)
axs.set_xlabel(L"$\gamma$",fontsize=14)
axs.set_xticks(ticks=range(0,howmany-1,5),labels=range(0,maxgamma,5),fontsize=10)
axs.set_yticks(ticks=range(0,howmany-1,5),labels=range(0,maxv0,5),fontsize=10)
# plt.colorbar(images[1],extend="both")
# axs.legend(labels1, [L"n=%$(Ns[1])",L"n=%$(Ns[2])",L"n=%$(Ns[3])",L"n=%$(Ns[4])",L"n=%$(Ns[5])",L"\sigma^{h2}=0",L"\sigma^{h3}=0"],loc="lower right",framealpha=1.0,fontsize=10)
# axs.legend(labels1, [L"n=%$(Ns[1])",L"n=%$(Ns[2])",L"n=%$(Ns[3])",L"n=%$(Ns[4])",L"n=%$(Ns[5])"],loc="lower right",framealpha=1.0,fontsize=10)
axs.legend(labels1, [L"n=%$(Ns[1])",L"n=%$(Ns[2])",L"\sigma^{h2}=0"],loc="lower right",framealpha=1.0,fontsize=10)

# fig.suptitle(L"\Re(\sigma_n)=0,D_T=%$(D_T),\lambda=%$(λ)",fontsize=10)
display(fig)
# fig.savefig("linear_instabs_λ=$(λ)_D_T=$(D_T)_k.png")
# fig.savefig("linear_instabs_λ=$(λ)_D_T=$(D_T)_k_hydro2.eps",bbox_inches="tight")
close(fig)