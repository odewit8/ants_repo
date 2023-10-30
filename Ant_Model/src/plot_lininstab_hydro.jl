using LinearAlgebra
using PyPlot
using LaTeXStrings
using Statistics
using Printf
rc("text", usetex=false) 
rc("mathtext", fontset="cm")
rc("font", family="serif", serif="cmr10", size=12)
rc("axes.formatter", use_mathtext = true)

# D_T = 1e-4
# D_R = 1.0
# D = 1.0
# η = 1.0
# α = 1.0
L = 1.0
# D_T = 0.01
# D_R = 1.0
# D = 1.0
# η = 1.0
# α = 1.0
sc = 2*pi/L
howmany = 100
# maxgamma = 510.0
# maxv0 = 15.1
maxgamma = 500
maxv0 = 12.5
D_T = 1.0
D_R = 1.0
D = 1.0
η = 1.0
α = 1.0
L = 1.0
sc = 2*pi/L
# howmany = 100
# maxgamma = 5.0
# maxv0 = 0.5
λ = 0.03
vs = range(0.0,maxv0,howmany)
γs = range(0.0,maxgamma,howmany)
heatmat = zeros(howmany,howmany)

function sigma(v0,D_T,D_R,γ,η,D,α,λ)
    return -4*pi^2*(v0^2/(2*D_R)+D_T)+4*pi^2*v0*γ*η/(2*D_R*(4*pi^2*D+α))+16*pi^4*γ*λ*v0^2*η/(16*D_R^2*(4*pi^2*D+α))
end

for vi ∈ 1:size(vs)[1], γi ∈ 1:size(γs)[1]
    # heatmat[vi,γi] = (4*pi^2/(2*D_R))*(-vs[vi]^2+vs[vi]*γs[γi]*η/(4*pi^2*D+α)+4*pi^2*vs[vi]^2*γs[γi]*λ*η/(8*D_R*(4*pi^2*D+α)))
    heatmat[vi,γi] = sigma(vs[vi],D_T,D_R,γs[γi],η,D,α,λ)
end

heatmat0 = zeros(howmany,howmany)
for vi ∈ 1:size(vs)[1], γi ∈ 1:size(γs)[1]
    # heatmat0[vi,γi] = (4*pi^2/(2*D_R))*(-vs[vi]^2+vs[vi]*γs[γi]*η/(4*pi^2*D+α))
    # heatmat0[vi,γi] = sigma(vs[vi],D_T,D_R,γs[γi],η,D,α,0.0)
    heatmat0[vi,γi] = sigma(vs[vi],D_T,D_R,γs[γi],η,D,α,0)
end

heatmatdiff = zeros(howmany,howmany)
for vi ∈ 1:size(vs)[1], γi ∈ 1:size(γs)[1]
    heatmatdiff[vi,γi] = sigma(vs[vi],D_T,D_R,γs[γi],η,D,α,λ)-sigma(vs[vi],D_T,D_R,γs[γi],η,D,α,0)
end

fig, axs = plt.subplots(1, 1, figsize=(6,4))
images = []

labels1 = []

which_N = 1
# D_T = 0.01

fig.suptitle(L"$\sigma,D_T=%$(D_T),D_R=%$(D_R),D=%$(D),\alpha=\eta=%$(η),\lambda=%$(λ)$",fontsize=10)
# push!(images,axs.imshow(heatmatdiff,origin="lower",cmap="viridis"))
push!(images,axs.imshow(heatmat,origin="lower",cmap="viridis"))
# push!(images,axs.imshow(heatmat0,origin="lower",cmap="viridis",vmin=0,vmax=10))
# push!(images,axs.imshow(imheatmats[1],origin="lower",cmap="viridis"))
push!(images,axs.contour(heatmat,levels=[0.0],colors="limegreen",linewidths=2))
h,_ = images[2].legend_elements()
push!(labels1,h[1])
push!(images,axs.contour(heatmat0,levels=[0.0],colors="yellow",linestyles="dashed",linewidths=2))
h,_ = images[3].legend_elements()
push!(labels1,h[1])
# push!(images,axs.contour(heatmatdiff,levels=[0.0],colors="seagreen",linestyles="dashed",linewidths=2))
# h,_ = images[4].legend_elements()
# push!(labels1,h[1])
# axs.legend(labels1, [L"\sigma=0",L"\sigma_{\lambda=0}=0",L"\sigma-\sigma_{\lambda=0}=0"],loc="upper center",framealpha=1.0,fontsize=10)
# axs.legend(labels1, [L"\sigma=0",L"\sigma_{\lambda=0}=0"],loc="upper center",framealpha=1.0,fontsize=10)
axs.legend(labels1, [L"\lambda=%$(λ)",L"\lambda=0"],loc="upper center",framealpha=1.0,fontsize=10)
axs.set_ylabel(L"$v_0$",fontsize=14)
axs.set_xlabel(L"$\gamma$",fontsize=14)
axs.set_xticks(ticks=range(0,howmany-1,5),labels=range(0,maxgamma,5),fontsize=10)
axs.set_yticks(ticks=range(0,howmany-1,5),labels=range(0,maxv0,5),fontsize=10)
plt.colorbar(images[1])
plt.tight_layout()
display(fig)
# fig.savefig("linear_instabs_hydro_λ=$(λ)_D_R=$(D_R).eps")
close(fig)