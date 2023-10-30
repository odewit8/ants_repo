using Distributed

@everywhere include("/home/od275/antmodel/Ant_Model/src/antmodel_pde.jl")
# @everywhere include("/home/od275/antmodel/Ant_Model/src/python_plots.jl")
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

params = dict_list(Dict(
    :phi => 0.0, 
    :Tf => 4.0,
    :Model => :modelParaEllplusLA,
    :Δt => 1e-5,
    :v0 => [1.0,6.0],
    :DR => 1.0,
    :DT => 0.01,
    :D => 1.0,
    :α => 1.0,
    :η => 1.0,
    :γ => 300, 
    :Nx => 31,
    :Ny => 31,
    :Nθ => 21,
    :λ1 => 3.1, #0.93,
    :ϵ => 0.0, 
    :IC => ["circle7"],
    :IC_chem => "chem0",
    :δ => 0.0,
    # :seed1 => 1,
    :saveplot => true
)) .|> (p->MyParams(; pairs(p)...))


iout = pmap(run_or_load_active_pde, params);
k=2
@unpack γ, DT, λ1, Nx, Ny, Tf, IC, v0 = params[k]
@unpack x, y, θ, pdesim, Δx, Δy, Δθ, phi = iout[k][1];
@unpack F, C, Rho, P, T = pdesim;

println(IC)
println(v0)

x = x .+ 0.5;
y = y .+ 0.5;
θ = θ .+ pi;
phi = 1;

Cos2θ = cos.(2*θ); Sin2θ = sin.(2*θ);

xx = [x̃ for x̃ ∈ x, ỹ ∈ y]'
yy = [ỹ for x̃ ∈ x, ỹ ∈ y]'

t = 1
ts = [1,4,6,50]
# close(fig)
fig, axs = plt.subplots(2, size(ts)[1], figsize=(12,6))
images = []

ts2 = [round((t-1)*(Tf/49),digits=2) for t ∈ ts]
colmap = PyPlot.plt.cm.viridis
# fig.suptitle(L"$\gamma$=%$(γ), $\lambda$=%$(round(λ1*Δx,digits=3)), $v_0$=%$(v0), times=%$(ts2)",fontsize=18)
Pn0 = (sqrt.(P[ts,1,:,:].^2 .+ P[ts,2,:,:].^2));
norm2 = colors2.SymLogNorm(linthresh=1.0,linscale=3.0,vmin=0.0,vmax=20.0);
for j ∈ 1:size(ts)[1]
    # norm=colors2.LogNorm()
    # push!(images,axs[1,j].imshow(Rho[ts[j],:,:],interpolation="quadric",norm=colors2.SymLogNorm(linthresh=1.0,linscale=3.0,vmin=0.0,vmax=20.0)))
    push!(images,axs[1,j].contourf(xx,yy,Rho[ts[j],:,:],levels=[0,0.1,0.5,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0,1.05,1.1,1.15,1.2,1.25,1.3,3.0,6.0,12.0,20.0],norm=colors2.SymLogNorm(linthresh=1.0,linscale=3.0,vmin=0.0,vmax=20.0),extend="max"))
    axs[1,j].set_xticks([])
    # axs[1,j].set_yticks([])
    axs[1,j].set_title(L"t=%$(ts2[j])")
    # Pn = (sqrt.(P[ts[j],2,end:-1:1,:].^2 .+ P[ts[j],1,end:-1:1,:].^2));
    Pn = (sqrt.(P[ts[j],1,:,:].^2 .+ P[ts[j],2,:,:].^2));
    # push!(images, axs[2,j].streamplot(xx, yy, (P[ts[j],2,end:-1:1,:]), (-P[ts[j],1,end:-1:1,:]), color=Pn, norm = norm2, cmap=colmap, maxlength=0.3,density=1.8,linewidth=1.0))
    # push!(images, axs[2,j].streamplot(xx, yy, (P[ts[j],2,end:-1:1,:]), (-P[ts[j],1,end:-1:1,:]), color=Pn, norm = norm2, cmap=colmap,density=1.5,linewidth=1.0))
    push!(images, axs[2,j].streamplot(xx, yy, (P[ts[j],2,:,:]), (P[ts[j],1,:,:]), color=Pn, norm = norm2, cmap=colmap,density=1.5,linewidth=1.0))
    # plt.colorbar(images[2*(j-1)+2])
    #plt.colorbar(images)
    
end

for j ∈ 1:4
    axs[1,j].set_aspect("equal")
    axs[2,j].set_aspect("equal")
    axs[1,j].set_xlim(0.0,1.0)
    axs[1,j].set_ylim(0.0,1.0)
    axs[2,j].set_xlim(0.0,1.0)
    axs[2,j].set_ylim(0.0,1.0)
end

for j ∈ 2:4
    # axs[2,j].set_xticks([])
    axs[2,j].set_yticks([])
    axs[1,j].set_yticks([])
    # axs[1,j].set_xlim(0.0,1.0)
    # axs[1,j].set_ylim(0.0,1.0)
end

axs[1,1].set_yticks([0,0.25,0.5,0.75,1.0])
axs[2,1].set_yticks([0,0.25,0.5,0.75,1.0])

fig.subplots_adjust(top=0.92,right=0.93,left=0.03,wspace=0.1,hspace=0.1)
cbar_ax_rho = fig.add_axes([0.95, 0.58, 0.01, 0.30])
cbar_ax_p = fig.add_axes([0.95, 0.15, 0.01, 0.30])
fig.colorbar(images[7], cax=cbar_ax_rho,extend="max",ticks=[0,1,10])
fig.colorbar(images[7], cax=cbar_ax_p,extend="max",ticks=[0,1,10])
# axs[1,1].set_ylabel(L"$\rho$",rotation=0,fontsize=16,loc="center",labelpad=12.0)
# axs[2,1].set_ylabel(L"$\mathbf{p}$",rotation=0,fontsize=16,loc="center",labelpad=12.0)
# axs[1,1].text(-3.5,15.0,L"$\rho$",fontsize=16)
axs[1,1].text(-0.3,0.5,L"$\rho$",fontsize=16)
axs[2,1].text(-0.3,0.5,L"$\mathbf{p}$",fontsize=16)
# plt.tight_layout()
display(fig)
# fig.savefig("paper_images/rho_P_λ=0.1_γ=$(γ)_$(IC)_v0=$(v0).png",bbox_inches="tight")
# fig.savefig("paper_images/rho_P_λ=0.1_γ=$(γ)_$(IC)_v0=$(v0).eps",bbox_inches="tight")
close(fig)

k=3
@unpack γ, DT, λ1, Nx, Ny, Nθ, Tf, IC, v0 = params[k]
@unpack x, y, θ, pdesim, Δx, Δy, Δθ, phi = iout[k][1];
@unpack F, C, Rho, P, T = pdesim;
println(DT)

# x = x .+ 0.5;
y = y .+ 0.5;
θ = θ .+ pi;
phi = 1;
θ2, X = np.meshgrid(θ,x)

fig, axs = plt.subplots(1, 1, figsize=(20,6))
images = []
push!(images,axs.imshow(Rho[3,:,:]))
display(fig)
close(fig)

Fend = F[end,:,:,:]
findmax(Fend)
findmax(Rho[end,:,:])
# fig, axs = plt.subplots(1, 1, figsize=(20,6))
fig, axs = plt.subplots(1, 1, figsize=(8,6))
# push!(images,axs[1].plot(θ,Fend[15,30,:]))
# axs[1].set_xticks([0,pi/2,pi,3*pi/2,2*pi])
# axs[1].set_xticklabels([L"0",L"\pi/2", L"\pi", L"3\pi/2",L"2\pi"])
push!(images,axs.plot(θ,Fend[15,30,:]))
axs.set_xticks([0,pi/2,pi,3*pi/2,2*pi])
# axs.set_xticklabels([L"0",L"\pi/2", L"\pi", L"3\pi/2",L"2\pi"])
axs.set_xticklabels([L"0",L"\pi/2", L"\pi", L"3\pi/2",L"2\pi"])
# axs[1].set_title(L"x_{\rho_{max}}")
# push!(images,axs[2].plot(θ,Fend[14,30,:]))
# axs[2].set_xticks([0,pi,2*pi])
# axs[2].set_xticklabels([L"0", L"\pi", L"2\pi"])
# axs[2].set_title(L"x_{\rho_{max}}+\Delta x")
# push!(images,axs[3].plot(θ,Fend[13,30,:]))
# axs[3].set_xticks([0,pi,2*pi])
# axs[3].set_xticklabels([L"0", L"\pi", L"2\pi"])
# axs[3].set_title(L"x_{\rho_{max}}+2\Delta x")
# fig.suptitle(L"f(t=5.0,x,y_{\rho_{max}},\theta),\gamma=%$(γ),\mathrm{Pe}=%$(v0),D_T=%$(DT),N_\theta=%$(Nθ)")
fig.suptitle(L"f(t=5.0,x_{\rho_{max}},y_{\rho_{max}},\theta),\gamma=%$(γ),\mathrm{Pe}=%$(v0),D_T=%$(DT),N_\theta=%$(Nθ)")
display(fig)
fig.savefig("paper_images/rho_x_maxs_P_λ=0.1_γ=$(γ)_$(IC)_v0=$(v0)_Nth=$(Nθ).png")

k=1
@unpack γ, DT, λ1, Nx, Ny, Nθ, Tf, IC, v0 = params[k]
@unpack x, y, θ, pdesim, Δx, Δy, Δθ, phi = iout[k][1];
@unpack F, C, Rho, P, T = pdesim;
println(DT)

# x = x .+ 0.5;
y = y .+ 0.5;
θ = θ .+ pi;
phi = 1;
θ2, X = np.meshgrid(θ,x)

fig, axs = plt.subplots(1, 1, figsize=(6,6))
images = []
# push!(images,axs.imshow(us21,origin="lower",vmin=-1.0,vmax=1.0))
Fyend = F[end,:,15,:]
Fyend2 = zeros(Nx,Nθ+1)
for j in 1:Nx
    for i in 1:Nθ
        Fyend2[j,mod(i+5,Nθ)+1] = Fyend[j,i]
    end
end
Fyend2[:,end] = Fyend2[:,1]
push!(images,axs.imshow(Fyend2,origin="lower",norm=colors2.SymLogNorm(linthresh=1.0,linscale=3.0,vmin=0.0,vmax=20.0),aspect=0.6))
axs.set_yticks(range(0,Nx-1,5))
axs.set_yticklabels(range(0,1.0,5))
axs.set_xticks(range(0,Nθ,5))
axs.set_xticklabels([L"$0\pi$",L"$\frac{1}{2}\pi$",L"$\pi$",L"$\frac{3}{2}\pi$",L"$2\pi$"])
axs.set_ylabel(L"$y$")
axs.set_xlabel(L"$\theta$")
plt.colorbar(images[1],fraction=0.046, pad=0.04,shrink=0.7,extend="max")
display(fig)
fig.savefig("paper_images/rho_y_maxs_λ=0.1_γ=$(γ)_$(IC)_v0=$(v0)_Nth=$(Nθ)_1.png",bbox_inches="tight")
fig.savefig("paper_images/rho_y_maxs_λ=0.1_γ=$(γ)_$(IC)_v0=$(v0)_Nth=$(Nθ)_1.eps",bbox_inches="tight")
close(fig)