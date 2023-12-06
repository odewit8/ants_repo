using Distributed
@everywhere include(srcdir("antmodel_pde.jl"))
@everywhere include(srcdir("accessory","config.jl"))

using PyPlot
using LaTeXStrings
using PyCall
@pyimport numpy as np
@pyimport matplotlib.colors as colors2
rc("text", usetex=false) 
rc("mathtext", fontset="cm")
rc("font", family="serif", serif="cmr10", size=12)
rc("axes.formatter", use_mathtext = true)

params = dict_list(Dict(
    :Tf => 2.0,
    :Model => :modelParaEllplusLA,
    :Δt => 1e-5,
    :v0 => [1.5,3.5],
    :DR => 1.0,
    :DT => 0.01,
    :D => 1.0,
    :α => 1.0,
    :η => 1.0,
    :γ => 325, 
    :Nx => 31,
    :Ny => 31,
    :Nθ => 21,
    :λ1 => 0.1,
    :IC => "rand_unif_seed",
    :seed1 => 4472,
    :saveplot => true
)) .|> (p->MyParams(; pairs(p)...))

iout = pmap(run_or_load_active_pde, params);

k=2
@unpack γ, DT, λ1, Nx, Ny, Tf, IC, v0, seed1 = params[k]
@unpack x, y, θ, pdesim, Δx, Δy, Δθ = iout[k][1];
@unpack F, C, Rho, P, T = pdesim;

x1 = x;
y1 = y;
θ1 = θ;

Cos2θ = cos.(2*θ); Sin2θ = sin.(2*θ);

xθ = [x̃ for x̃ ∈ x1, θ̃ ∈ θ]
θx = [θ̃ for x̃ ∈ x1, θ̃ ∈ θ]
xx = [x̃ for x̃ ∈ x1, ỹ ∈ y1]
yy = [ỹ for x̃ ∈ x1, ỹ ∈ y1]
x2, y2 = np.meshgrid(x, y, indexing = "ij", sparse = false)


Pend = P[end,:,:,:]
t = 1
ts = [1,3,5,50]

fig, axs = plt.subplots(2, size(ts)[1], figsize=(13,7))
images = []

Pnend = (sqrt.(P[end,1,:,:]'.^2 .+ P[end,2,:,:]'.^2));

ts2 = [round((t-1)*(Tf/49),digits=1) for t ∈ ts]
colmap = PyPlot.plt.cm.viridis
norm2 = colors2.SymLogNorm(linthresh=1.0,linscale=3.0,vmin=0.0,vmax=20.0);
for j ∈ 1:size(ts)[1]
    push!(images,axs[1,j].contourf(xx,yy,Rho[ts[j],:,:],levels=20,extend="max"))
    axs[1,j].set_xticks([])
    axs[1,j].set_title(L"t=%$(ts2[j])")
    Pn = (sqrt.(P[ts[j],1,:,:]'.^2 .+ P[ts[j],2,:,:]'.^2));
    lw = 0.15.+2.0*Pn./findmax(Pnend)[1]
    push!(images, axs[2,j].streamplot(x2', y2', P[ts[j],1,:,:]', P[ts[j],2,:,:]', color=Pn, cmap=colmap, linewidth=lw,density=1.2))
end

for j ∈ 1:4
    axs[1,j].set_aspect("equal")
    axs[2,j].set_aspect("equal")
    axs[1,j].set_xlim(x[1],x[end])
    axs[1,j].set_ylim(y[1],y[end])
    axs[2,j].set_xlim(x[1],x[end])
    axs[2,j].set_ylim(y[1],y[end])
end

for j ∈ 2:4
    # axs[2,j].set_xticks([])
    axs[2,j].set_yticks([])
    axs[1,j].set_yticks([])
    # axs[1,j].set_xlim(0.0,1.0)
    # axs[1,j].set_ylim(0.0,1.0)
end

tickfontsize = 13
labelfontsize = 18
axs[2,1].set_xticks([x1[1],-0.25,0.0,0.25,x1[end]])
axs[2,1].set_xticklabels([-0.5,-0.25,0.0,0.25,0.5],fontsize=tickfontsize)
axs[2,1].set_xlabel(L"x",fontsize=labelfontsize)
axs[2,1].set_yticks([x1[1],-0.25,0.0,0.25,x1[end]])
axs[2,1].set_yticklabels([-0.5,-0.25,0.0,0.25,0.5],fontsize=tickfontsize)
axs[2,1].set_ylabel(L"y",fontsize=labelfontsize)
axs[1,1].set_yticks([x1[1],-0.25,0.0,0.25,x1[end]])
axs[1,1].set_yticklabels([-0.5,-0.25,0.0,0.25,0.5],fontsize=tickfontsize)
axs[2,2].set_xticks([x1[1],-0.25,0.0,0.25,x1[end]])
axs[2,2].set_xticklabels([-0.5,-0.25,0.0,0.25,0.5],fontsize=tickfontsize)
axs[2,3].set_xticks([x1[1],-0.25,0.0,0.25,x1[end]])
axs[2,3].set_xticklabels([-0.5,-0.25,0.0,0.25,0.5],fontsize=tickfontsize)
axs[2,4].set_xticks([x1[1],-0.25,0.0,0.25,x1[end]])
axs[2,4].set_xticklabels([-0.5,-0.25,0.0,0.25,0.5],fontsize=tickfontsize)

fig.subplots_adjust(top=0.92,right=0.93,left=0.03,wspace=0.1,hspace=0.1)
cbar_ax_rho = fig.add_axes([0.95, 0.58, 0.01, 0.30])
cbar_ax_p = fig.add_axes([0.95, 0.15, 0.01, 0.30])
cbar_ax_rho1 = fig.colorbar(images[1], cax=cbar_ax_rho,extend="max",shrink=0.9)
cbar_ax_p1 = fig.colorbar(images[2].lines, cax=cbar_ax_p,extend="max",shrink=0.9)
cbar_ax_rho1.ax.set_title(L"\rho",fontsize=labelfontsize)
cbar_ax_p1.ax.set_title(L"|\mathbf{p}|",fontsize=labelfontsize)
display(fig)
# fig.savefig("paper_images/rho_P_λ=0.1_γ=$(γ)_$(IC)_seed=$(seed1)_v0=$(v0).png",bbox_inches="tight")
# fig.savefig("paper_images/rho_P_λ=0.1_γ=$(γ)_$(IC)_seed=$(seed1)_v0=$(v0).eps",bbox_inches="tight")
close(fig)