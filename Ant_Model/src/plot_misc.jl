using Distributed


@everywhere using DrWatson
@everywhere begin
        quickactivate(@__DIR__, "Ant_Model")
        using Parameters
        include(srcdir("antmodel_pde.jl"))
        include(srcdir("accessory","config.jl"))
        include(srcdir("python_plots.jl"))
        include(srcdir("oscar_plots.jl"))
end


params = dict_list(Dict(
    :phi => 0.0, 
    :Tf => 8.0,
    :Model => :modelParaEllplusLA, 
    :Δt => 1e-5,
    :v0 => [0.1:0.2:2.0...],
    :DR => 0.01,
    :DT => 0.01,
    :D => 0.1,
    :α => 40.0,
    :η => 40.0,
    :γ => [0.4:0.5:5.0...], 
    :Nx => 31,
    :Ny => 31,
    :Nθ => 21,
    :λ1 => 3.0,
    :ϵ => 0.0, 
    :IC => "rand_unif",
    :IC_chem => "chem0",
    :δ => 0.0,
    :saveplot => true
)) .|> (p->MyParams(; pairs(p)...))

# params = dict_list(Dict(
#     :phi => 0.0, 
#     :Tf => 6.0,
#     :Model => :modelParaEllplusLA, 
#     :Δt => 1e-5,
#     :v0 => [0.1:1.5:15.1...],
#     :DR => 1.0,
#     :DT => 0.0001,
#     :D => 1.0,
#     :α => 1.0,
#     :η => 1.0,
#     :γ => [10.0:50.0:510.0...], 
#     :Nx => 31,
#     :Ny => 31,
#     :Nθ => 21,
#     :λ1 => 3.1,
#     :ϵ => 0.0, 
#     :IC => "rand_unif",
#     :IC_chem => "chem0",
#     :δ => 0.0,
#     :saveplot => true
# )) .|> (p->MyParams(; pairs(p)...))

# params = dict_list(Dict(
#     :phi => 0.0, 
#     :Tf => 5.0,
#     :Model => :modelParaEllplusLA, 
#     :Δt => 1e-5,
#     :v0 => [1:2:10...],
#     :DR => 1.0,
#     :DT => 0.0001,
#     :D => 1.0,
#     :α => 1.0,
#     :η => 1.0,
#     :γ => [10:150:630...], 
#     :Nx => 31,
#     :Ny => 31,
#     :Nθ => 21,
#     :λ1 => 0.5,
#     :ϵ => 0.0, 
#     :IC => "rand_unif",
#     :IC_chem => "chem0",
#     :δ => 0.0,
#     #:seed1 => [346,287,385,974,658,361, 499, 486],
#     :saveplot => true
# )) .|> (p->MyParams(; pairs(p)...))

# params = dict_list(Dict(
#     :phi => 0.0, 
#     :Tf => 5.0,
#     :Model => :modelParaEllplusLA, 
#     :Δt => 1e-5,
#     :v0 => [1:2:10...],
#     :DR => 1.0,
#     :DT => 0.01,
#     :D => 1.0,
#     :α => 1.0,
#     :η => 1.0,
#     :γ => [10:150:630...], 
#     :Nx => 31,
#     :Ny => 31,
#     :Nθ => 21,
#     :λ1 => 3.1,
#     :ϵ => 0.0, 
#     :IC => "rand_unif",
#     :IC_chem => "chem0",
#     :δ => 0.0,
#     # :seed1 => 1,
#     :saveplot => true
# )) .|> (p->MyParams(; pairs(p)...))

# params = dict_list(Dict(
#     :phi => 0.0, 
#     :Tf => 3.0,
#     :Model => :modelParaEll, #:modelParaEllplusLA,
#     :Δt => 1e-5,
#     :v0 => 3,
#     :DR => 1.0,
#     :DT => 1.0,
#     :D => 1.0,
#     :α => 1.0,
#     :η => 1.0,
#     :γ => 250, 
#     :Nx => 31,
#     :Ny => 31,
#     :Nθ => 21,
#     :λ1 => 0.0, #0.93,
#     :ϵ => 0.0, 
#     :IC => "Pert3",
#     :IC_chem => "chem0",
#     :δ => 0.1,
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

# for k in [10.0:50.0:510.0...]
#     println(k)
# end

using PyPlot
using Statistics
using PyCall
@pyimport matplotlib.colors as colors2
rc("text", usetex=false) 
rc("mathtext", fontset="cm")
rc("font", family="serif", serif="cmr10", size=12)
rc("axes.formatter", use_mathtext = true)

function polarisation2(f::Array{T,3}, Cos2θ::Array{T,1}, Sin2θ::Array{T,1}; Δθ::Float64)::Array{T,3} where T<:Real
    sizes = size(f);
    p2 = Array{T,3}(undef, 2, sizes[1], sizes[2])

    for i = 1:sizes[1]
        for j = 1:sizes[2]
            p2[1,i,j] = sum(f[i,j,:].*Cos2θ) * Δθ
            p2[2,i,j] = sum(f[i,j,:].*Sin2θ) * Δθ
        end
    end

    return p2
end

function get_plotlims(A)
    lmax = maximum(A);
    lmin = minimum(A);

    # return (lmin, lmax)
    return (lmin, min(1.0, lmax))
end

ip = 1
@unpack γ, λ1, v0, Nx, Ny, Nθ = params[ip]
@unpack x, y, θ, pdesim, Δx, Δθ, Δy = iout[ip][1];
@unpack F, C, Rho, P, T = pdesim;

x1 = x .+ 0.5;
y1 = y .+ 0.5;
xx = [x̃ for x̃ ∈ x1, ỹ ∈ y1]'
yy = [ỹ for x̃ ∈ x1, ỹ ∈ y1]'

Cos2θ = cos.(2*θ); Sin2θ = sin.(2*θ);
Cosθ = cos.(θ); Sinθ = sin.(θ);

# F0 = Array{Float64,3}(undef,Nx, Ny, Nθ)
# Fend = F[1,:,:,:]

# for ix ∈ 1:Nx
#     for iy ∈ 1:Ny
#         for iθ ∈ 1:Nθ
#             ixx = (((ix-1)+6) % Nx) + 1
#             iθ2 = (((iθ - 1)+0) % Nθ) + 1
#             # println(ixx)
#             # println(iθ2)
#             F0[ix,iy,iθ] = Fend[ixx,iy,iθ2]
#         end
#     end
# end

# save_object("F0_DR=1.0_DT=0.0001_Nx=30_Nθ=20_v0=9.1_γ=410.0_2.jld2",F0)
# t=1
# F0 = F[t,:,:,:]
# Rho0 = sum(F0, dims = 3) * Δθ
# P = polarisation(F0, Cosθ, Sinθ, Δθ = Δθ)

ip = 25
@unpack γ, λ1, v0, Nx, Ny, Nθ = params[ip]
println(γ,v0,λ1)
@unpack x, y, θ, pdesim, Δx, Δθ, Δy = iout[ip][1];
@unpack F, C, Rho, P, T = pdesim;

fig, axs = plt.subplots(1, 1, figsize=(8,6))
images = []
t=50
# P2 = polarisation2(F[t,:,:,:],Cos2θ,Sin2θ,Δθ=Δθ)
sum(P[end,:,:,:],dims=(2,3))
# sum(P2[:,:,:],dims=(2,3))
Fend = F[end,:,:,:]
findmax(Fend)
findmax(Rho[end,:,:])
push!(images,axs.plot(θ,Fend[10,16,:]))
axs.set_xticks([0,pi,2*pi])
axs.set_xticklabels([L"0", L"\pi", L"2\pi"])
fig.suptitle(L"f(t=5.0,x_{\rho_{max}},y_{\rho_{max}},\theta),\gamma=610,v_0=9,D_T=10^{-2}")
display(fig)
fig.savefig("f_rhomax_DT=0.01_v0=9_γ=610_rand_unif_Tf=5_lambda=0.1.eps")


# sum(Rho[t,:,:])*Δx^2
# C2 = C[t,:,:]
# C3 = 0.5*(C2+transpose(C2))
push!(images,axs.imshow(Rho[t,:,:],interpolation="bilinear"))
# push!(images,axs.imshow(C2-C3,interpolation="bilinear"))
plt.colorbar(images[1])
axs.set_xticks([])
axs.set_yticks([])
fig.suptitle(L"\rho")
display(fig)
fig, axs = plt.subplots(1, 1, figsize=(8,6))
colmap = PyPlot.plt.cm.viridis_r
images = []
Pn = (sqrt.(P[t,1,end:-1:1,:].^2 .+ P[t,2,end:-1:1,:].^2));
plims = get_plotlims(Pn);
norm1 = matplotlib.colors.Normalize(vmin=plims[1], vmax= plims[2])
push!(images, axs.streamplot(xx, yy, (P[t,2,end:-1:1,:]), (-P[t,1,end:-1:1,:]), color=Pn, norm = norm1, cmap=colmap))
display(fig)
fig, axs = plt.subplots(1, 1, figsize=(8,6))
colmap = PyPlot.plt.cm.viridis_r
images = []
Pn = (sqrt.(P2[1,end:-1:1,:].^2 .+ P2[2,end:-1:1,:].^2));
plims = get_plotlims(Pn);
norm1 = matplotlib.colors.Normalize(vmin=plims[1], vmax= plims[2])
push!(images, axs.streamplot(xx, yy, (P2[2,end:-1:1,:]), (-P2[1,end:-1:1,:]), color=Pn, norm = norm1, cmap=colmap))
display(fig)

colmap = PyPlot.plt.cm.viridis_r
fig, axs = plt.subplots(2, 3, figsize=(24,16))
images= []
F0 = F[1,:,:,:]
Rho0 = sum(F0, dims = 3) * Δθ
P = polarisation(F0, Cosθ, Sinθ, Δθ = Δθ)
push!(images,axs[1,1].imshow(Rho0[:,:,1],interpolation="bilinear"))
Pn = (sqrt.(P[1,end:-1:1,:].^2 .+ P[2,end:-1:1,:].^2));
plims = get_plotlims(Pn);
norm1 = matplotlib.colors.Normalize(vmin=plims[1], vmax= plims[2])
push!(images, axs[2,1].streamplot(xx, yy, (P[2,end:-1:1,:]), (-P[1,end:-1:1,:]), color=Pn, norm = norm1, cmap=colmap))
F0 = F[3,:,:,:]
Rho0 = sum(F0, dims = 3) * Δθ
P = polarisation(F0, Cosθ, Sinθ, Δθ = Δθ)
push!(images,axs[1,2].imshow(Rho0[:,:,1],interpolation="bilinear"))
Pn = (sqrt.(P[1,end:-1:1,:].^2 .+ P[2,end:-1:1,:].^2));
plims = get_plotlims(Pn);
norm1 = matplotlib.colors.Normalize(vmin=plims[1], vmax= plims[2])
push!(images, axs[2,2].streamplot(xx, yy, (P[2,end:-1:1,:]), (-P[1,end:-1:1,:]), color=Pn, norm = norm1, cmap=colmap))
F0 = F[20,:,:,:]
Rho0 = sum(F0, dims = 3) * Δθ
P = polarisation(F0, Cosθ, Sinθ, Δθ = Δθ)
push!(images,axs[1,3].imshow(Rho0[:,:,1],interpolation="bilinear"))
Pn = (sqrt.(P[1,end:-1:1,:].^2 .+ P[2,end:-1:1,:].^2));
plims = get_plotlims(Pn);
norm1 = matplotlib.colors.Normalize(vmin=plims[1], vmax= plims[2])
push!(images, axs[2,3].streamplot(xx, yy, (P[2,end:-1:1,:]), (-P[1,end:-1:1,:]), color=Pn, norm = norm1, cmap=colmap))
# Pn2 = (sqrt.(P2[1,end:-1:1,:].^2 .+ P2[2,end:-1:1,:].^2));
# p2lims = get_plotlims(Pn2);
# norm2 = matplotlib.colors.Normalize(vmin=p2lims[1], vmax= p2lims[2])
# push!(images, axs.streamplot(xx, yy, (P2[2,end:-1:1,:]), (-P2[1,end:-1:1,:]), color=Pn, norm = norm2, cmap=colmap))
for j ∈ 1:3
    axs[1,j].set_aspect("equal")
    axs[2,j].set_aspect("equal")
    axs[2,j].set_xticks([])
    axs[2,j].set_yticks([])
    axs[1,j].set_xticks([])
    axs[1,j].set_yticks([])
    # axs[1,j].set_xlim(0.0,1.0)
    # axs[1,j].set_ylim(0.0,1.0)
    axs[2,j].set_xlim(0.0,1.0)
    axs[2,j].set_ylim(0.0,1.0)
end
# axs[1,1].set_ylabel(L"$\rho$",rotation=0,fontsize=16,loc="center",labelpad=12.0)
# axs[2,1].set_ylabel(L"$\mathbf{p}$",rotation=0,fontsize=16,loc="center",labelpad=12.0)
display(fig)

plot(θ,F[end,1,5,:])
plt.imshow(Rho[end,:,:])
plt.savefig("plot__2.png")

ix = 10
jx = 10
fig, axs = plt.subplots(ix, jx, figsize=(12,10))
images = []
t=50

for i ∈ 1:ix
    for j ∈ 1:jx
        k = ix*(i-1)+j
        @unpack γ, λ1, v0 = params[k]
        @unpack pdesim, Δx = iout[k][1];
        @unpack F, C, Rho, P, T = pdesim;
        # push!(images,axs[ix+1-i,j].imshow(Rho[t,:,:],interpolation="bilinear"))
        push!(images,axs[ix+1-i,j].imshow(Rho[end,:,:],norm=colors2.LogNorm(),vmin=0.001,vmax=100.0,interpolation="bilinear"))
        axs[ix+1-i,j].set_xticks([])
        axs[ix+1-i,j].set_yticks([])
        # axs[ix+1-i,j].set_title(L"$\gamma=%$(γ),v_0=%$(v0)$",fontsize=8)
        if j == 1
            if mod(v0+0.5,2) == 0
                axs[ix+1-i,j].set_ylabel(L"\mathrm{Pe}=%$(v0)",fontsize=8)
            end
        end
        if ix+1-i == ix
            if mod(γ+25,100) == 0
                axs[ix+1-i,j].set_xlabel(L"\gamma=%$(γ)",fontsize=8)
            end
        end
        # plt.colorbar(images[k],ax=axs[ix+1-i,j],fraction=0.046, pad=0.04)
    end
end
# fig.suptitle(L"\rho(t=5.0,x,y),D_T=10^{-2},D_R=1.0,D=\alpha=\eta=1.0,\lambda=0.1,\Delta t=10^{-5},N_x=N_y=31,N_\theta=21")
fig.subplots_adjust(top=0.9,bottom=0.1,right=0.9,left=0.1,wspace=0.4,hspace=0.4)
# cbar_ax = fig.add_axes([0.95, 0.25, 0.01, 0.5])
# fig.colorbar(images[1], cax=cbar_ax)
# fig.supxlabel(L"\gamma")
# fig.supylabel(L"\mathrm{Pe}")
# axs.set_xlabel(L"\gamma")
cb = fig.colorbar(images[5],ax=axs,shrink=0.6)
# fig.axes[end].set(xlabel=L"\rho(t=5.0,\mathbf{x})")
cb.ax.set_title(L"\rho")
display(fig)
fig.savefig("paper_images/rhos_DT=0.01_Pe=10_γ=500_rand_unif_Tf=5_lambda=0.png",bbox_inched="tight")
fig.savefig("paper_images/rhos_DT=0.01_Pe=10_γ=500_rand_unif_Tf=5_lambda=0.eps",bbox_inched="tight")

fig, axs = plt.subplots(1, 1, figsize=(8,5))
images = []
ix =11
Γ = zeros(ix,ix)
V0s = zeros(ix,ix)
L2s = zeros(ix,ix)
for i ∈ 1:ix
    for j ∈ 1:ix
        k = ix*(j-1)+i
        @unpack γ, λ1, v0 = params[k]
        @unpack pdesim, Δx = iout[k][1];
        @unpack F, C, Rho, P, T = pdesim;
        Γ[i,j] = γ
        V0s[i,j] = v0
        L2s[i,j] = sqrt(sum(Rho[end,:,:].^2))*Δx
    end
end
xs = range(0,550,step=50)
ys = xs./(4*pi^2+1)
push!(images,axs.scatter(x=Γ[:],y=V0s[:],c=L2s[:],cmap="viridis"))
push!(images,axs.plot(xs,ys,label=L"\frac{\gamma}{4\pi^2+1}=v_0"))
axs.set_xlabel(L"\gamma")
axs.set_ylabel(L"v_0")
axs.set_xlim(0,520)
axs.set_ylim(0,16)
axs.legend(loc="upper left",framealpha=1.0,fontsize=16)
plt.colorbar(images[1])
fig.suptitle(L"\|\rho(t=6.0)\|_{L^2},D_T=10^{-4},D_R=1.0,D=\alpha=\eta=1.0,\lambda=0.0,\Delta t=10^{-5},N_x=N_y=31,N_\theta=21",fontsize=10)
display(fig)
fig.savefig("L2_invisc_instab_v0=0.1-15.1_γ=10-510_rand_unif_Tf=6_lambda=0.0.png")
fig.savefig("L2_invisc_instab_v0=0.1-15.1_γ=10-510_rand_unif_Tf=6_lambda=0.0.eps")



# @unpack γ, λ1, v0, IC = params[1]
# @unpack pdesim, Δx = iout[1][1];
# fig.suptitle(L"$\rho(t=5.0,x),\lambda=%$(round(Δx*λ1,digits=3))$",fontsize=18)
@unpack Nx, Ny, IC, Tf,λ1 = params[1]
@unpack x, y, θ, pdesim, Δθ, Δx = iout[1][1];

fig.subplots_adjust(top=0.92,right=0.93,left=0.03,wspace=0.001,hspace=0.1)
cbar_ax = fig.add_axes([0.95, 0.25, 0.01, 0.5])
fig.colorbar(images[1], cax=cbar_ax)

fig.suptitle(L"\rho(T_f,x),\lambda=%$(round(Δx*λ1,digits=3))",fontsize=20)
# plt.tight_layout()
display(fig)
fig.savefig("rhos_v0=0.1-2.0_γ=0.4-5.0_$(IC)_Tf=$(Tf).png")
fig.savefig("rhos_v0=0.1-2.0_γ=0.4-5.0_$(IC)_Tf=$(Tf).eps")
close(fig)


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

fig, axs = plt.subplots(1, 1, figsize=(8,5))
images = []
tsf = [t for t ∈ 1:50]

ix2 = ix*ix
for j ∈ 1:ix2
    @unpack γ, λ1, v0 = params[j]
    @unpack pdesim, Δx = iout[j][1];
    @unpack F, C, Rho, P, T = pdesim;
    P2 = Array{Float64,4}(undef, 50, 2, Nx, Ny)
    for ti ∈ 1:50
        P2[ti,:,:,:] = polarisation2(F[ti,:,:,:], Cos2θ, Sin2θ, Δθ = Δθ)
    end
    intP2s = [norm(mean(P2[t,:,:,:],dims=[2,3])) for t ∈ tsf]
    # push!(images,axs.plot(tsf,intP2s,label=L"\gamma=%$(γ),v_0=%$(v0)"))
    push!(images,axs.plot(tsf,intP2s,color="black"))
end

# println("γ=$(γ),λ=$(round(Δx*λ1,digits=3)),v_0=$(v0)")
# plt.legend(loc="upper left",fontsize=8)
fig.suptitle(L"|P_2(t)|,\lambda=%$(round(Δx*λ1,digits=3))", fontsize=12)
axs.set_xlabel(L"t")
axs.set_ylim(0.3,0.8)
axs.set_xlim(30,50)
# axs.set_xticks(tticks,tticks2)
display(fig)
fig.savefig("int_p2_v0=0.1-2.0_γ=0.4-5.0_$(IC)_Tf=$(Tf).png")
fig.savefig("int_p2_v0=0.1-2.0_γ=0.4-5.0_$(IC)_Tf=$(Tf)_high_end.eps")

close(fig)

fig, axs = plt.subplots(1, 1, figsize=(8,5))
images = []
ix=11
Γ = zeros(ix,ix)
V0s = zeros(ix,ix)
P2s = zeros(ix,ix)
for i ∈ 1:ix
    for j ∈ 1:ix
        k = ix*(j-1)+i
        @unpack γ, λ1, v0 = params[k]
        @unpack pdesim, Δx = iout[k][1];
        @unpack F, C, Rho, P, T = pdesim;
        P2 = Array{Float64,4}(undef, 50, 2, Nx, Ny)
        for ti ∈ 1:50
            P2[ti,:,:,:] = polarisation2(F[ti,:,:,:], Cos2θ, Sin2θ, Δθ = Δθ)
        end
        intP2s = [norm(mean(P2[t,:,:,:],dims=[2,3])) for t ∈ 1:50]
        # println(intP2s[end])
        # push!(images,axs.scatter(x=γ,y=v0,color=intP2s[end]))
        Γ[i,j] = γ
        V0s[i,j] = v0
        P2s[i,j] = intP2s[end]
        # if findmax(Rho[end,:,:])[1] <= 1.2
        #     push!(images,axs.scatter(x=γ,y=v0,color="black"))
        # elseif intP2s[end] >= 0.25
        #     push!(images,axs.scatter(x=γ,y=v0,color="blue"))
        # elseif intP2s[end] - intP2s[end-1] > 0.001
        #     push!(images,axs.scatter(x=γ,y=v0,color="grey"))
        # else
        #     push!(images,axs.scatter(x=γ,y=v0,color="red"))
        # end
    end
end

push!(images,axs.scatter(x=Γ[:],y=V0s[:],c=P2s[:],cmap="viridis"))
axs.set_xlabel(L"\gamma")
axs.set_ylabel(L"v_0")
axs.set_xlim(0,520)
axs.set_ylim(0,16)
plt.colorbar(images[1])
fig.suptitle(L"|P_2(T_f)|,\lambda=%$(round(Δx*λ1,digits=3))", fontsize=12)
display(fig)
fig.savefig("grid_P2_heatmap_v0=0.1-2.0_γ=0.4-5.0_$(IC)_Tf=$(Tf).png")
fig.savefig("grid_P2_heatmap_v0=0.1-2.0_γ=0.4-5.0_$(IC)_Tf=$(Tf).eps")

close(fig)