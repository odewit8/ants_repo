using DrWatson
quickactivate("Ant_Model")

using DataFrames
using PyPlot
using Printf
using PyCall
using Ranges
using Rotations
using JLD2
@pyimport matplotlib.animation as anim2
@pyimport matplotlib.colors as colors2

include(srcdir("accessory","config.jl"))

function plot_pert(Params::MyParams)

    fn = datadir("PDE_Sims", savename(params[1], "jld2"; sigdigits = 3, accesses = [:DR, :DT, :IC, :Model, :Tf, :phi, :v0, :δ]));
    sn_eps = plotsdir("Paperplots", replace(savename("pert", Params, "eps"; sigdigits = 3, accesses = [:DR, :DT, :IC, :IC_chem, :Model, :Tf, :phi, :v0, :δ]), "δ" => "d"))

    if !isfile(fn)
        println("Case $(savename(Params, "jld2"; sigdigits = 3, ignores = [:NumOut, :saveplot, :savedata])) does not exist!");
        return
    else
        Sim = wload(fn);
    end

    @unpack F̂norm, pdesim, Δx, Δy, Δθ = Sim;
    @unpack T = pdesim;

    dv = Δx*Δy*Δθ
    dv = 1; # plot rescaled quantity

    fig, ax = plt.subplots(1,1,constrained_layout=true,figsize=(3,2.1))
    ax.plot(T, F̂norm .* dv)

    ax.set_xlim([0, T[end]])
    ax.set_ylim(bottom=0)

    ax.set_xticks([0, T[end]/2, T[end]])
    ax.set_xlabel(L"t")
    ax.set_ylabel(L"e")

    display(fig)
    fig.savefig(sn_eps)

    close(fig)
end







function pyplot_2dpaper(Params::MyParams; num_times::Int64 = 3, save_over::Bool = true)

    output_times = [0.0, 0.5, 1.5];
    # fn = datadir("PDE_Sims", savename(Params, "jld2"; sigdigits = 3, ignores = [:NumOut, :saveplot, :savedata, :Nx, :Ny, :Nθ, :Δt]));
    fn = datadir("PDE_Sims4", savename(Params, "jld2"; sigdigits = 3, ignores = [:NumOut, :saveplot, :savedata]));
    sn_eps = plotsdir("plots_1", replace(savename(Params, 
    "png"; sigdigits = 3, accesses = [:δ, :phi, :v0, :Model, :IC, :IC_chem, :α, :η, :γ, :D, :DT, :Tf, :Nx]),
     "δ" => "d", "α" => "a", "η" => "n", "γ" => "g"))

    if isfile(sn_eps) && (!save_over)
        println("Figure already exists")
        return
    end

    if !isfile(fn)
        println("Case $(savename(Params, "jld2"; sigdigits = 3, ignores = [:NumOut, :saveplot, :savedata])) does not exist!");
        return
    else
        Sim = wload(fn);
    end

    @unpack saveplot, v0, phi = Params;
    @unpack x, y, θ, pdesim, Δx, Δy, Δθ, phi = Sim;
    @unpack F, C, Rho, P, T = pdesim;

    # shift to [0,1]^2 x [0, 2Pi]
    x = x .+ 0.5;
    y = y .+ 0.5;
    θ = θ .+ pi;
    phi = 1;

    xx = [x̃ for x̃ ∈ x, ỹ ∈ y]'
    yy = [ỹ for x̃ ∈ x, ỹ ∈ y]'

    if length(T)<= num_times
        idx = 1:length(T)
    else
        idx = Int.(round.(range(1, stop=length(T), length = num_times)));
    end
    # idx = idx[2:end]; #do not plot the initial time

    idx = [argmin(abs.(T .- output_times[j])) for j = 1:num_times]

    p1lims = get_plotlims(phi.*Rho[idx,:,:]);
    # println(p1lims)
    # mob = (Params.Model == :model2 ? phi.*Rho : max.(phi.*Rho.*(1 .- phi.*Rho), 0.0));

    # p3lims = get_plotlims(mob[idx,:,:]);
    # println(p3lims)
    Pn = (sqrt.(P[idx,1,:,:].^2 .+ P[idx,2,:,:].^2)./Rho[idx,:,:]);
    p2lims = get_plotlims(Pn);

    Nr = 2
    #Nr = 4
    Nc = length(idx)

    fig, axs = plt.subplots(Nr, Nc, constrained_layout=true, figsize=(12,8))
    # fig, axs = plt.subplots(Nr, Nc, figsize=(12,8), tight_layout = true)
    # norm1 = matplotlib.colors.Normalize(vmin=p1lims[1], vmax= p1lims[2]);
    # norm1 = matplotlib.colors.Normalize(vmin=0, vmax= 5);
    norm1 = matplotlib.colors.LogNorm(vmin=0.1, vmax=20);

    norm2 = matplotlib.colors.Normalize(vmin=p2lims[1], vmax= p2lims[2]);
    # norm3 = matplotlib.colors.Normalize(vmin=p3lims[1], vmax= p3lims[2]);

    images = []
    colmap = PyPlot.plt.cm.viridis_r

    for i = 1:Nc
        t = idx[i];

        push!(images, axs[1,i].contourf(x,y,Rho[t,:,:], norm = norm1, cmap=colmap))
        axs[1,i].set_title(L"t = %$(round(T[t],digits=1))")

        Pn = (sqrt.(P[t,1,:,:].^2 .+ P[t,2,:,:].^2)./Rho[t,:,:]);
        # lw = 5 .* Pn ./ maximum(Pn) # Line Widths
        push!(images, axs[2,i].streamplot(xx, yy, (P[t,2,:,:]./Rho[t,:,:]), (P[t,1,:,:]./Rho[t,:,:]), color=Pn, norm = norm2, cmap=colmap))
        # push!(images, axs[2,i].streamplot(xx, yy, P[i,1,:,:]', P[i,2,:,:]', density=0.6,color="k",linewidth=lw))

        # push!(images, axs[3,i].contourf(x,y,mob[t,:,:]',norm = norm3, cmap=colmap))
        # push!(images, axs[4,i].contourf(x,y,C[t,:,:]',cmap=colmap))

        # [axs[j,i].set_xlim([0, 1]) for j = 1:4]
        # [axs[j,i].set_ylim([0, 1]) for j = 1:4]
        # [axs[j,i].set_xticks([0, 0.5, 1]) for j = 1:4]
        # [axs[j,i].set_yticks([0, 0.5, 1]) for j = 1:4]
        # [axs[j,i].set_aspect("equal", adjustable = "box") for j = 1:4]
        # [axs[j,i].set_xlabel(L"x") for j = 1:4]
        # [axs[j,i].set_ylabel(L"y") for j = 1:4]
        # [axs[j,i].label_outer() for j = 1:4]
        [axs[j,i].set_xlim([0, 1]) for j = 1:2]
        [axs[j,i].set_ylim([0, 1]) for j = 1:2]
        [axs[j,i].set_xticks([0, 0.5, 1]) for j = 1:2]
        [axs[j,i].set_yticks([0, 0.5, 1]) for j = 1:2]
        [axs[j,i].set_aspect("equal", adjustable = "box") for j = 1:2]
        [axs[j,i].set_xlabel(L"x") for j = 1:2]
        [axs[j,i].set_ylabel(L"y") for j = 1:2]
        [axs[j,i].label_outer() for j = 1:2]


    end

    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm1, cmap = colmap), ax = axs[1,Nc])
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm2, cmap = colmap), ax = axs[2,Nc])
    # fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm3, cmap = colmap), ax = axs[3,Nc])
    # fig.colorbar(matplotlib.cm.ScalarMappable(cmap = colmap), ax = axs[4,Nc])

    # fig.subplots_adjust(top = 0.99, bottom = 0.11, left = 0.12, right = 0.99)
    axs[1,1].text(-0.3,.8,L"\rho",fontsize=16)
    axs[2,1].text(-0.3,.8,L"{\bf p}/\rho",fontsize=16)
    # axs[3,1].text(-0.3,.8,L"m_{11}",fontsize=16)
    # axs[4,1].text(-0.3,.8,L"c",fontsize=16)
    display(fig)
    fig.savefig(sn_eps)
    close(fig)

end

function get_plotlims(A)
    lmax = maximum(A);
    lmin = minimum(A);

    # return (lmin, lmax)
    return (lmin, min(1.0, lmax))
end


function pyplot_2dpaper2(Params::MyParams; num_times::Int64 = 1, save_over::Bool = true)

    output_times = [2.9];
    # fn = datadir("PDE_Sims", savename(Params, "jld2"; sigdigits = 3, ignores = [:NumOut, :saveplot, :savedata, :Nx, :Ny, :Nθ, :Δt]));
    fn = datadir("PDE_Sims4", savename(Params, "jld2"; sigdigits = 3, ignores = [:NumOut, :saveplot, :savedata]));
    sn_eps = plotsdir("plots_1", replace(savename(Params, 
    "png"; sigdigits = 3, accesses = [:δ, :phi, :v0, :Model, :IC, :IC_chem, :α, :η, :γ, :D, :DT, :Tf, :Nx]),
     "δ" => "d", "α" => "a", "η" => "n", "γ" => "g"))

    if isfile(sn_eps) && (!save_over)
        println("Figure already exists")
        return
    end

    if !isfile(fn)
        println("Case $(savename(Params, "jld2"; sigdigits = 3, ignores = [:NumOut, :saveplot, :savedata])) does not exist!");
        return
    else
        Sim = wload(fn);
    end

    @unpack saveplot, v0, phi, DT, DR, γ = Params;
    @unpack x, y, θ, pdesim, Δx, Δy, Δθ, phi = Sim;
    @unpack F, C, Rho, P, T = pdesim;

    # shift to [0,1]^2 x [0, 2Pi]
    x = x .+ 0.5;
    y = y .+ 0.5;
    θ = θ .+ pi;
    phi = 1;

    xx = [x̃ for x̃ ∈ x, ỹ ∈ y]'
    yy = [ỹ for x̃ ∈ x, ỹ ∈ y]'

    # idx = idx[2:end]; #do not plot the initial time

    idx = [argmin(abs.(T .- output_times[j])) for j = 1:num_times]

    p1lims = get_plotlims(phi.*Rho[idx,:,:]);
    # println(p1lims)
    # mob = (Params.Model == :model2 ? phi.*Rho : max.(phi.*Rho.*(1 .- phi.*Rho), 0.0));

    # p3lims = get_plotlims(mob[idx,:,:]);
    # println(p3lims)
    Pn = (sqrt.(P[idx,1,:,:].^2 .+ P[idx,2,:,:].^2)./Rho[idx,:,:]);
    p2lims = get_plotlims(Pn);

    Nr = 2
    #Nr = 4
    Nc = length(idx)
    Nc = 3

    fig, axs = plt.subplots(Nr, Nc, constrained_layout=true, figsize=(20,10))
    # fig, axs = plt.subplots(Nr, Nc, figsize=(12,8), tight_layout = true)
    # norm1 = matplotlib.colors.Normalize(vmin=p1lims[1], vmax= p1lims[2]);
    # norm1 = matplotlib.colors.Normalize(vmin=0, vmax= 5);
    norm1 = matplotlib.colors.LogNorm(vmin=0.1, vmax=20);

    norm2 = matplotlib.colors.Normalize(vmin=p2lims[1], vmax= p2lims[2]);
    # norm3 = matplotlib.colors.Normalize(vmin=p3lims[1], vmax= p3lims[2]);

    images = []
    colmap = PyPlot.plt.cm.viridis_r

    for i = 1:Nc
        t = idx[1];

        push!(images, axs[1,i].plot(θ,F[t,20-4*(i-1),20,:]))
        push!(images, axs[2,i].plot(x,Rho[idx[1],20-4*(i-1),:]))
        # axs[1,i].set_title(L"t = %$(round(T[t],digits=1))")

        # [axs[1,i].set_xlim([0, 1]) for j = 1:2]
        # [axs[1,i].set_ylim([0, 1]) for j = 1:2]
        # [axs[1,i].set_xticks([0, 0.5, 1]) for j = 1:2]
        # [axs[1,i].set_yticks([0, 0.5, 1]) for j = 1:2]
        # [axs[1,i].set_aspect("equal", adjustable = "box") for j = 1:2]
        axs[1,i].set_yscale("log")
        # axs[i].set_ylim([1e-3,1e2])
        axs[1,i].set_xlabel(L"\theta")
        axs[1,i].set_ylabel(L"f(x,\theta)")
        axs[2,i].set_yscale("log")
        axs[2,i].set_xlabel(L"y")
        axs[2,i].set_ylabel(L"\rho(\cdot,x)")
        # [axs[1,i].label_outer() for j = 1:2]


    end
    axs[1,1].set_title(L"D_T=%$(DT),D_R=%$(DR),v_0=%$(v0),\gamma=%$(γ)")

    # fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm1, cmap = colmap), ax = axs[1,Nc])
    # fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm2, cmap = colmap), ax = axs[2,Nc])

    # axs[1,1].text(-0.3,.8,L"\rho",fontsize=16)
    # axs[2,1].text(-0.3,.8,L"{\bf p}/\rho",fontsize=16)

    display(fig)
    fig.savefig(sn_eps)
    close(fig)

end



function pyplot_video0(Params::MyParams; final_time::Float64 = -1.0, save_over::Bool = true)


    fn = datadir("PDE_Sims", savename(Params, "jld2"; sigdigits = 3, ignores = [:NumOut, :saveplot, :savedata]));
    sn_mp4 = plotsdir("Papervideos", replace(savename(Params, "mp4"; sigdigits = 3, accesses = [:δ, :phi, :v0, :Model, :IC, :Tf, :Nx]), "δ" => "d"))

    if isfile(sn_mp4) && (!save_over)
        println("Video already exists")
        return
    end

    if !isfile(fn)
        println("Case $(savename(Params, "jld2"; sigdigits = 3, ignores = [:NumOut, :saveplot, :savedata])) does not exist!");
        return
    else
        Sim = wload(fn);
    end

    @unpack saveplot, v0, phi = Params;
    @unpack x, y, θ, pdesim, Δx, Δy, Δθ, phi = Sim;
    @unpack F, Rho, P, T = pdesim;

    # shift to [0,1]^2 x [0, 2Pi]
    x = x .+ 0.5;
    y = y .+ 0.5;
    θ = θ .+ pi;

    if final_time == -1
        idx = 1:length(T); # use all the points
    else
        idx = 1:argmin((T .- final_time).^2);
    end

    xx = [x̃ for x̃ ∈ x, ỹ ∈ y]'
    yy = [ỹ for x̃ ∈ x, ỹ ∈ y]'


    p1lims = get_plotlims(phi.*Rho[idx,:,:]);
    # println(p1lims)
    mob = (Params.Model == :model2 ? phi.*Rho : max.(phi.*Rho.*(1 .- phi.*Rho), 0.0));

    p3lims = get_plotlims(mob[idx,:,:]);
    # println(p3lims)
    Pn = (sqrt.(P[idx,1,:,:].^2 .+ P[idx,2,:,:].^2)./Rho[idx,:,:]);
    p2lims = get_plotlims(Pn);

    Fs = dropdims(sum(F, dims = 3),dims = 3)*Δy; # average in y
    p4lims = get_plotlims(phi.*Fs[idx,:,:])

    Nr = 2; Nc = 2;

#
    # fig, axs = plt.subplots(Nr, Nc, constrained_layout=true, figsize=(10,8))
    # fig = plt.figure()
    # ax1 = fig.add_subplot(121)
    # ax2 = fig.add_subplot(122)
    fig, (ax1, ax3, ax2, ax4) = plt.subplots(2,2, constrained_layout=true, figsize=(12,10), dpi=150)

    norm1 = matplotlib.colors.Normalize(vmin=p1lims[1], vmax= p1lims[2]);
    norm2 = matplotlib.colors.Normalize(vmin=p2lims[1], vmax= p2lims[2]);
    norm3 = matplotlib.colors.Normalize(vmin=p3lims[1], vmax= p3lims[2]);
    norm4 = matplotlib.colors.Normalize(vmin=p4lims[1], vmax= p4lims[2]);

    colmap = PyPlot.plt.cm.viridis_r

    # k=0,1,...,frames-1
    function plot_one(k)
        t = k + 1

        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()

        y1 = ax1.contourf(x,y,phi.*Rho[t,:,:]', norm = norm1, cmap=colmap)
        ax1.set_title(L"\rho", fontsize=16)
        fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm1, cmap = colmap), ax = ax1)

        y2 = ax2.contourf(x,θ,phi.*Fs[t,:,:]', norm = norm4, cmap=colmap)
        ax2.set_title(L"f", fontsize=16)
        fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm4, cmap = colmap), ax = ax2)

        Pn = (sqrt.(P[t,1,:,:].^2 .+ P[t,2,:,:].^2)./Rho[t,:,:])';
        # lw = 5 .* Pn ./ maximum(Pn) # Line Widths
        y3 = ax3.streamplot(xx, yy, (P[t,1,:,:]./Rho[t,:,:])', (P[t,2,:,:]./Rho[t,:,:])', color=Pn, norm = norm2, cmap=colmap)
        fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm2, cmap = colmap), ax = ax3)
        ax3.set_title(L"{\bf q}", fontsize=16)
        # push!(images, axs[2,i].streamplot(xx, yy, P[i,1,:,:]', P[i,2,:,:]', density=0.6,color="k",linewidth=lw))

        y4 = ax4.contourf(x,y,mob[t,:,:]',norm = norm3, cmap=colmap)
        fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm3, cmap = colmap), ax = ax4)
        ax4.set_title(L"m_{11}", fontsize=16)

        ax1.set_xlim([0, 1]); ax2.set_xlim([0, 1]); ax3.set_xlim([0, 1]); ax4.set_xlim([0, 1]);
        ax1.set_ylim([0, 1]); ax2.set_ylim([0, 2*pi]); ax3.set_ylim([0, 1]); ax4.set_ylim([0, 1]);
        ax1.set_xticks([0, 0.5, 1]); ax2.set_xticks([0, 0.5, 1]); ax3.set_xticks([0, 0.5, 1]); ax4.set_xticks([0, 0.5, 1]);
        ax1.set_yticks([0, 0.5, 1]); ax2.set_yticks([0, pi, 2*pi]); ax3.set_yticks([0, 0.5, 1]); ax4.set_yticks([0, 0.5, 1]);
        ax2.set_yticklabels(["0", L"\pi", L"2\pi"])

        ax1.set_aspect("equal", adjustable = "box"); ax2.set_aspect(1/(2*pi), adjustable = "box");
        ax3.set_aspect("equal", adjustable = "box"); ax4.set_aspect("equal", adjustable = "box");
        ax1.set_xlabel(L"x", fontsize=16); ax2.set_xlabel(L"x", fontsize=16); ax3.set_xlabel(L"x", fontsize=16); ax4.set_xlabel(L"x", fontsize=16);
        ax1.set_ylabel(L"y", fontsize=16); ax2.set_ylabel(L"\theta", fontsize=16); ax3.set_ylabel(L"y", fontsize=16); ax4.set_ylabel(L"y", fontsize=16);

        plt.suptitle(L"t = %$(round(T[t],digits=3))", fontsize=18)

        plot()
    #     display(fig)
    #     close(fig)
    # #     plt.show()
    end

    function init()
        plot_one(1)
    end

    interval=200 # milli seconds

    # mywriter = anim.FFMpegWriter()
    FFwriter=anim.FFMpegWriter(fps=30, extra_args=["-vcodec", "libx264"], bitrate=-1)
    myanim = anim.FuncAnimation(fig, plot_one, frames=length(idx), init_func = init, interval = interval)

    println(typeof(myanim))

    myanim.save(sn_mp4, writer = FFwriter)
    close(fig)

end

function pyplot_video(Params::MyParams; final_time::Float64 = -1.0, save_over::Bool = true)


    fn = datadir("PDE_Sims", savename(Params, "jld2"; sigdigits = 3, ignores = [:NumOut, :saveplot, :savedata]));
    sn_mp4 = plotsdir("Papervideos", replace(savename(Params, "mp4"; sigdigits = 3, accesses = [:δ, :phi, :v0, :Model, :IC, :Tf, :Nx]), "δ" => "d"))

    if isfile(sn_mp4) && (!save_over)
        println("Video already exists")
        return
    end

    if !isfile(fn)
        println("Case $(savename(Params, "jld2"; sigdigits = 3, ignores = [:NumOut, :saveplot, :savedata])) does not exist!");
        return
    else
        Sim = wload(fn);
    end

    @unpack saveplot, v0, phi = Params;
    @unpack x, y, θ, pdesim, Δx, Δy, Δθ, phi = Sim;
    @unpack F, Rho, C, P, T = pdesim;

    # shift to [0,1]^2 x [0, 2Pi]
    x = x .+ 0.5;
    y = y .+ 0.5;
    θ = θ .+ pi;

    if final_time == -1
        idx = 1:length(T); # use all the points
    else
        idx = 1:argmin((T .- final_time).^2);
    end

    xx = [x̃ for x̃ ∈ x, ỹ ∈ y]'
    yy = [ỹ for x̃ ∈ x, ỹ ∈ y]'


    p1lims = get_plotlims(phi.*Rho[idx,:,:]);
    # println(p1lims)
    mob = (Params.Model == :model2 ? phi.*Rho : max.(phi.*Rho.*(1 .- phi.*Rho), 0.0));

    p3lims = get_plotlims(mob[idx,:,:]);
    # println(p3lims)
    Pn = (sqrt.(P[idx,1,:,:].^2 .+ P[idx,2,:,:].^2)./Rho[idx,:,:]);
    p2lims = get_plotlims(Pn);

    Fs = dropdims(sum(F, dims = 3),dims = 3)*Δy; # average in y
    p4lims = get_plotlims(phi.*Fs[idx,:,:])

    Nr = 2; Nc = 2;

    fig, (ax1, ax3, ax2, ax4) = plt.subplots(2,2, constrained_layout=true, figsize=(12,10), dpi=150)

    norm1 = matplotlib.colors.Normalize(vmin=p1lims[1], vmax= p1lims[2]);
    norm2 = matplotlib.colors.Normalize(vmin=p2lims[1], vmax= p2lims[2]);
    norm3 = matplotlib.colors.Normalize(vmin=p3lims[1], vmax= p3lims[2]);
    norm4 = matplotlib.colors.Normalize(vmin=p4lims[1], vmax= p4lims[2]);

    colmap = PyPlot.plt.cm.viridis_r

    # initial plot
    t = 1;

    p1 = [ax1.contourf(x,y,phi.*Rho[t,:,:]', norm = norm1, cmap=colmap)]
    ax1.set_title(L"\rho", fontsize=16)
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm1, cmap = colmap), ax = ax1)

    p2 = [ax2.contourf(x,θ,phi.*Fs[t,:,:]', norm = norm4, cmap=colmap)]
    ax2.set_title(L"f", fontsize=16)
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm4, cmap = colmap), ax = ax2)

    Pn = (sqrt.(P[t,1,:,:].^2 .+ P[t,2,:,:].^2)./Rho[t,:,:])';
    # lw = 5 .* Pn ./ maximum(Pn) # Line Widths
    y3 = ax3.streamplot(xx, yy, (P[t,1,:,:]./Rho[t,:,:])', (P[t,2,:,:]./Rho[t,:,:])', color=Pn, norm = norm2, cmap=colmap)
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm2, cmap = colmap), ax = ax3)
    ax3.set_title(L"{\bf q}", fontsize=16)
    # push!(images, axs[2,i].streamplot(xx, yy, P[i,1,:,:]', P[i,2,:,:]', density=0.6,color="k",linewidth=lw))

    p4 = [ax4.contourf(x,y,mob[t,:,:]',norm = norm3, cmap=colmap)]
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm3, cmap = colmap), ax = ax4)
    ax4.set_title(L"m_{11}", fontsize=16)

    ax1.set_xlim([0, 1]); ax2.set_xlim([0, 1]); ax3.set_xlim([0, 1]); ax4.set_xlim([0, 1]);
    ax1.set_ylim([0, 1]); ax2.set_ylim([0, 2*pi]); ax3.set_ylim([0, 1]); ax4.set_ylim([0, 1]);
    ax1.set_xticks([0, 0.5, 1]); ax2.set_xticks([0, 0.5, 1]); ax3.set_xticks([0, 0.5, 1]); ax4.set_xticks([0, 0.5, 1]);
    ax1.set_yticks([0, 0.5, 1]); ax2.set_yticks([0, pi, 2*pi]); ax3.set_yticks([0, 0.5, 1]); ax4.set_yticks([0, 0.5, 1]);
    ax2.set_yticklabels(["0", L"\pi", L"2\pi"])

    ax1.set_aspect("equal", adjustable = "box"); ax2.set_aspect(1/(2*pi), adjustable = "box");
    ax3.set_aspect("equal", adjustable = "box"); ax4.set_aspect("equal", adjustable = "box");
    ax1.set_xlabel(L"x", fontsize=16); ax2.set_xlabel(L"x", fontsize=16); ax3.set_xlabel(L"x", fontsize=16); ax4.set_xlabel(L"x", fontsize=16);
    ax1.set_ylabel(L"y", fontsize=16); ax2.set_ylabel(L"\theta", fontsize=16); ax3.set_ylabel(L"y", fontsize=16); ax4.set_ylabel(L"y", fontsize=16);

    plt.suptitle(L"t = %$(round(T[t],digits=3))", fontsize=18)

    # k=0,1,...,frames-1
    function update(k)
        t = k + 1

        for tp in p1[1].collections
            tp.remove()
        end
        for tp in p2[1].collections
            tp.remove()
        end
        for tp in p4[1].collections
            tp.remove()
        end

        p1[1] = ax1.contourf(x,y,phi.*Rho[t,:,:]', norm = norm1, cmap=colmap)
        p2[1] = ax2.contourf(x,θ,phi.*Fs[t,:,:]', norm = norm4, cmap=colmap)
        p4[1] = ax4.contourf(x,y,mob[t,:,:]',norm = norm3, cmap=colmap)

        ax3.clear()
        Pn = (sqrt.(P[t,1,:,:].^2 .+ P[t,2,:,:].^2)./Rho[t,:,:])';
        # lw = 5 .* Pn ./ maximum(Pn) # Line Widths
        ax3.streamplot(xx, yy, (P[t,1,:,:]./Rho[t,:,:])', (P[t,2,:,:]./Rho[t,:,:])', color=Pn, norm = norm2, cmap=colmap)
        fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm2, cmap = colmap), ax = ax3)
        ax3.set_title(L"{\bf q}", fontsize=16)
        ax3.set_xlim([0, 1]);
        ax3.set_ylim([0, 1]);
        ax3.set_xticks([0, 0.5, 1]);
        ax3.set_yticks([0, 0.5, 1]);
        ax3.set_aspect("equal", adjustable = "box");
        ax3.set_xlabel(L"x", fontsize=16);
        ax3.set_ylabel(L"y", fontsize=16);
        
        plt.suptitle(L"t = %$(round(T[t],digits=3))", fontsize=18)
        plot()

    end

    # mywriter = anim.FFMpegWriter()
    plt.close()
    FFwriter=anim.FFMpegWriter(fps=30, extra_args=["-vcodec", "libx264"], bitrate=-1)
    myanim = anim.FuncAnimation(fig, update, frames=length(idx), interval = 10, repeat=true)

    println(typeof(myanim))

    myanim.save(sn_mp4, writer = FFwriter)
    close(fig)
end


function test_video()

    x = [0:0.01:2pi;]
    fig, axs = plt.subplots(1,1)

    function one_plot(t)
        clf()
        plot(sin.(x .+ t/10.0))
        plot()
    end

    function init()
        one_plot(0)
    end

    interval=200 # milli seconds

    myanim = anim.FuncAnimation(fig, one_plot, frames=3, init_func = init, interval = interval, blit = true)

    println(typeof(myanim))

    myanim.save(plotsdir("Papervideos","test.mp4"), fps=10)
end


function pyplot_slides(Params::MyParams; final_time::Float64 = -1.0, save_over::Bool = true)

    fn = datadir("PDE_Sims", savename(Params, "jld2"; sigdigits = 3, ignores = [:NumOut, :saveplot, :savedata]));
    save_dir = plotsdir("Papervideos", replace(savename(Params; sigdigits = 3, accesses = [:δ, :phi, :v0, :Model, :IC, :Tf, :Nx]), "δ" => "d"))

    if ispath(save_dir) && (!save_over)
        println("Video already exists")
        return
    elseif !ispath(save_dir)
        mkpath(save_dir)
    end

    if !isfile(fn)
        println("Case $(savename(Params, "jld2"; sigdigits = 3, ignores = [:NumOut, :saveplot, :savedata])) does not exist!");
        return
    else
        Sim = wload(fn);
    end

    @unpack saveplot, v0, phi = Params;
    @unpack x, y, θ, pdesim, Δx, Δy, Δθ, phi = Sim;
    @unpack F, Rho, P, T = pdesim;

    # shift to [0,1]^2 x [0, 2Pi]
    x = x .+ 0.5;
    y = y .+ 0.5;
    θ = θ .+ pi;

    if final_time == -1
        idx = 1:length(T); # use all the points
    else
        idx = 1:argmin((T .- final_time).^2);
    end

    xx = [x̃ for x̃ ∈ x, ỹ ∈ y]'
    yy = [ỹ for x̃ ∈ x, ỹ ∈ y]'

    p1lims = get_plotlims(phi.*Rho[idx,:,:]);
    mob = (Params.Model == :model2 ? phi.*Rho : max.(phi.*Rho.*(1 .- phi.*Rho), 0.0));

    p3lims = get_plotlims(mob[idx,:,:]);
    Pn = (sqrt.(P[idx,1,:,:].^2 .+ P[idx,2,:,:].^2)./Rho[idx,:,:]);
    p2lims = get_plotlims(Pn);

    Fs = dropdims(sum(F, dims = 3),dims = 3)*Δy; # average in y
    p4lims = get_plotlims(phi.*Fs[idx,:,:])

    norm1 = matplotlib.colors.Normalize(vmin=p1lims[1], vmax= p1lims[2]);
    norm2 = matplotlib.colors.Normalize(vmin=p2lims[1], vmax= p2lims[2]);
    norm3 = matplotlib.colors.Normalize(vmin=p3lims[1], vmax= p3lims[2]);
    norm4 = matplotlib.colors.Normalize(vmin=p4lims[1], vmax= p4lims[2]);

    colmap = PyPlot.plt.cm.viridis_r

    for t ∈ idx

        fig, axs = plt.subplots(2,2, constrained_layout=true, figsize=(12,10), dpi=300)
        fig_name = @sprintf "/frame%03i.png" t
        axs[1,1].contourf(x,y,phi.*Rho[t,:,:]', norm = norm1, cmap=colmap)
        axs[1,1].set_title(L"\rho", fontsize=16)
        fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm1, cmap = colmap), ax = axs[1,1])

        axs[1,2].contourf(x,θ,phi.*Fs[t,:,:]', norm = norm4, cmap=colmap)
        axs[1,2].set_title(L"f", fontsize=16)
        fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm4, cmap = colmap), ax = axs[1,2])

        Pn = (sqrt.(P[t,1,:,:].^2 .+ P[t,2,:,:].^2)./Rho[t,:,:])';
        # lw = 5 .* Pn ./ maximum(Pn) # Line Widths
        axs[2,1].streamplot(xx, yy, (P[t,1,:,:]./Rho[t,:,:])', (P[t,2,:,:]./Rho[t,:,:])', color=Pn, norm = norm2, cmap=colmap)
        fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm2, cmap = colmap), ax = axs[2,1])
        axs[2,1].set_title(L"{\bf q}", fontsize=16)
        # push!(images, axs[2,i].streamplot(xx, yy, P[i,1,:,:]', P[i,2,:,:]', density=0.6,color="k",linewidth=lw))

        axs[2,2].contourf(x,y,mob[t,:,:]',norm = norm3, cmap=colmap)
        fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm3, cmap = colmap), ax = axs[2,2])
        axs[2,2].set_title(L"m_{11}", fontsize=16)

        [axs[i,j].set_xlim([0, 1]) for i = 1:2, j = 1:2]
        [axs[i,j].set_ylim([0, 1]) for i = 1:2, j = 1:2]
        axs[1,2].set_ylim([0, 2*pi])
        [axs[i,j].set_xticks([0, 0.5, 1]) for i = 1:2, j = 1:2]
        [axs[i,j].set_yticks([0, 0.5, 1]) for i = 1:2, j = 1:2]
        axs[1,2].set_yticks([0, pi, 2*pi])
        axs[1,2].set_yticklabels(["0", L"\pi", L"2\pi"])

        [axs[i,j].set_aspect("equal", adjustable = "box") for i = 1:2, j = 1:2]
        axs[1,2].set_aspect(1/(2*pi), adjustable = "box");

        [axs[i,j].set_xlabel(L"x", fontsize=16) for i = 1:2, j = 1:2]
        [axs[i,j].set_ylabel(L"y", fontsize=16) for i = 1:2, j = 1:2]
        axs[1,2].set_ylabel(L"\theta", fontsize=16);

        plt.suptitle(L"t = %$(round(T[t],digits=3))", fontsize=18)
        # display(fig)
        fig.savefig(save_dir*fig_name)
        close(fig)
    end

end


function test_video()

    x = [0:0.01:2pi;]
    fig, axs = plt.subplots(1,1)

    function one_plot(t)
        clf()
        plot(sin.(x .+ t/10.0))
        plot()
    end

    function init()
        one_plot(0)
    end

    interval=200 # milli seconds

    myanim = anim.FuncAnimation(fig, one_plot, frames=3, init_func = init, interval = interval, blit = true)

    println(typeof(myanim))

    myanim.save(plotsdir("Papervideos","test.mp4"), fps=10)
end

function pyplot_2dpaper_stat(Params::MyParams; num_times::Int64 = 1, save_over::Bool = true)

    output_times = [1.9];
    # fn = datadir("PDE_Sims", savename(Params, "jld2"; sigdigits = 3, ignores = [:NumOut, :saveplot, :savedata, :Nx, :Ny, :Nθ, :Δt]));
    fn = datadir("PDE_Sims4", savename(Params, "jld2"; sigdigits = 3, ignores = [:NumOut, :saveplot, :savedata]));
    sn_eps = plotsdir("plots_1", replace(savename(Params, 
    "png"; sigdigits = 3, accesses = [:δ, :phi, :v0, :Model, :IC, :IC_chem, :α, :η, :γ, :D, :DT, :Tf, :Nx]),
     "δ" => "d", "α" => "a", "η" => "n", "γ" => "g"))

    if isfile(sn_eps) && (!save_over)
        println("Figure already exists")
        return
    end

    if !isfile(fn)
        println("Case $(savename(Params, "jld2"; sigdigits = 3, ignores = [:NumOut, :saveplot, :savedata])) does not exist!");
        return
    else
        Sim = wload(fn);
    end

    @unpack saveplot, v0, phi, DT, DR, γ, Nx, Nθ, Ny = Params;
    @unpack x, y, θ, pdesim, Δx, Δy, Δθ, phi = Sim;
    @unpack F, C, Rho, P, T = pdesim;
    Pe = v0

    # G4 = load_object("F4_DR=0.01_DT=0.01_Nx=61_Nθ=41_v0=0.5_γ=1.0.jld2")
    G4 = load_object("G6.jld2")
    rho4 = sum(G4, dims = 3) * Δθ
    # C4 = load_object("C4_DR=0.01_DT=0.01_Nx=61_Nθ=41_v0=0.5_γ=1.0.jld2")
    C4 = load_object("C6.jld2")
    P4 = load_object("P6.jld2")

    # shift to [0,1]^2 x [0, 2Pi]
    x = x .+ 0.5;
    y = y .+ 0.5;
    θ2 = θ;
    θ = θ .+ pi;
    phi = 1;

    xx = [x̃ for x̃ ∈ x, ỹ ∈ y]'
    yy = [ỹ for x̃ ∈ x, ỹ ∈ y]'

    # idx = idx[2:end]; #do not plot the initial time

    idx = [argmin(abs.(T .- output_times[j])) for j = 1:num_times]

    p1lims = get_plotlims(phi.*Rho[idx,:,:]);
    # println(p1lims)
    # mob = (Params.Model == :model2 ? phi.*Rho : max.(phi.*Rho.*(1 .- phi.*Rho), 0.0));

    # p3lims = get_plotlims(mob[idx,:,:]);
    # println(p3lims)
    Pn = (sqrt.(P[idx,1,:,:].^2 .+ P[idx,2,:,:].^2)./Rho[idx,:,:]);
    p2lims = get_plotlims(Pn);

    Nr = 1
    #Nr = 4
    Nc = length(idx)
    Nc = 1

    fig, axs = plt.subplots(Nr, Nc, constrained_layout=true, figsize=(10,8))

    images = []
    colmap = PyPlot.plt.cm.viridis_r

    npoints = 100
    rs = [sqrt((x̃-0.5)^2+(ỹ-0.5)^2) for x̃∈x,ỹ∈y]
    # rs2 = LinRange(0, 0.5*sqrt(2),npoints)
    
    rs2 = LinRange(0, 0.5,npoints)
    p_rs = [(RotMatrix{2}(-atan(y[j]-0.5,x[i]-0.5))*(P4[:,i,j]))[1] for i∈1:Nx,j∈1:Ny]

    prs2 = []

    IJs = [(i-21)^2 + (j-21)^2 for i∈1:Nx,j∈1:Ny]
    IJs = [j for j∈IJs]
    IJs = sort(unique(IJs))
    IJs_sqrt_dx = [sqrt(j)*Δx for j ∈ IJs]

    function pr_avg(r)
        sum1 = 0
        if r ∈ IJs
            for i∈1:Nx,j∈1:Ny
                s = (i-21)^2 + (j-21)^2
                if s == r
                    if (i-21 == 0) && (j-21 == 0)
                        sum1 += p_rs[i,j]
                    elseif abs(i-21) == abs(j-21)
                        # println("same abs not 0")
                        sum1 += (p_rs[i,j])/4
                    else
                        # println("diff abs not 0")
                        sum1 += (p_rs[i,j])/8
                    end
                end
            end
        
        end
        return sum1
    end

    # prs3 = [pr_avg(r) for r ∈ IJs]
    # # prs3 = sort(unique(prs3))
    # println(size(IJs))
    # for i ∈ 1:689
    #     @printf("pr_avg1 = %f, prs3 = %f,r^2 = %d",pr_avg(IJs[i]),prs3[i],IJs[i])
    #     println(" ")
    # end

    function int_p_r(r)
        sum1 = 0
        N = 0
        for i∈1:Nx,j∈1:Ny
            s = sqrt((x[i]-0.5)^2 + (y[j]-0.5)^2)
            if s == 0
                sum1 += p_rs[i,j]/(2*0.3826*Δx)
                N += 1
            elseif 0 < s <= r
                sum1 += p_rs[i,j]/(2*s)
                N += 1
            end
        end
        return (r^2/N)*sum1
    end

    println(int_p_r(0))

    function int_p_r2(r)
        sum = 0
        for i∈1:Nx,j∈1:Ny
            s = sqrt((x[i]-0.5)^2 + (y[j]-0.5)^2)
            if 0 < s <= r
                # if (abs(x[i]-0.5) == 0) && (abs(y[j]-0.5) == 0)
                #     sum += p_rs[i,j]
                # elseif abs(x[i]-0.5) == abs(y[j]-0.5)
                #     sum += (p_rs[i,j])/4
                if (i-21 == 0) && (j-21 == 0)
                    sum += p_rs[i,j]
                elseif abs(i-21) == abs(j-21)
                    # println("same abs not 0")
                    sum += (p_rs[i,j])/4
                else
                    # println("diff abs not 0")
                    sum += (p_rs[i,j])/8
                end
            end
        end
        return sum*(pi*r^2*Δx^2)
    end
    
    # print(int_p_r(0.1))
    # for k ∈ 1:npoints
    #     # println(rs2[k])
    #     push!(prs2,[])
    #     for i∈1:Nx,j∈1:Ny
    #         if rs2[k] <= rs[i,j] < rs2[k+1]
    #             push!(prs2[k],p_rs[i,j])
    #         end
    #     end
    # end

    # prs3 = zeros(npoints)
    # for k ∈ 1:npoints
    #     # println(size(prs2[k]))
    #     prs3[k] = mean(prs2[k])
    # end

    # println(prs3)

    # for i∈1:Nx
    #     println(round.(rho4[i,:],digits=1))
    # end
    # println()
    # for i∈1:Nx
    #     println(P4[:,i,21])
    # end
    # println()
    # println(findmax(rho4))
    # p
    


    # center(x,y)=[18, 21]
    # center rho (x,y)=[20,21]
    
    # fig.suptitle(L"D_T=%$(DT),D_R=%$(DR),v_0=%$(v0),\gamma=%$(γ), centre \ (x,y)=(21,21), r=\sqrt{13}\Delta x", fontsize=16)
    # push!(images, axs[1,1].plot(θ,G4[21+2,21+3,:]))
    # axs[1,1].set_xlabel(L"\theta")
    # axs[1,1].set_ylabel(L"f(x,y,\theta)")
    # axs[1,1].set_title(L"(x,y)=(23,24), (r,\phi)=(\sqrt{13},%$(round(atan(3/2)/pi,digits=2)) \pi)")
    # axs[1,1].set_ylim(0,14)
    # push!(images, axs[1,2].plot(θ,G4[21+2,21-3,:]))
    # axs[1,2].set_ylim(0,14)
    # axs[1,2].set_title(L"(x,y)=(23,18), (r,\phi)=(\sqrt{13},%$(round(2+atan(-3/2)/pi,digits=2)) \pi)")
    # push!(images, axs[1,3].plot(θ,G4[21-2,21-3,:]))
    # axs[1,3].set_ylim(0,14)
    # axs[1,3].set_title(L"(x,y)=(19,18), (r,\phi)=(\sqrt{13},%$(round(2 + atan(3/2)/pi - 1,digits=2)) \pi)")
    # push!(images, axs[1,4].plot(θ,G4[21-2,21+3,:]))
    # axs[1,4].set_ylim(0,14)
    # axs[1,4].set_title(L"(x,y)=(19,24), (r,\phi)=(\sqrt{13},%$(round(1 - atan(3/2)/pi,digits=2)) \pi)")
    # push!(images, axs[2,1].plot(θ,G4[21+3,21+2,:]))
    # axs[2,1].set_ylim(0,14)
    # axs[2,1].set_title(L"(x,y)=(24,23), (r,\phi)=(\sqrt{13},%$(round(atan(2/3)/pi,digits=2)) \pi)")
    # push!(images, axs[2,2].plot(θ,G4[21+3,21-2,:]))
    # axs[2,2].set_ylim(0,14)
    # axs[2,2].set_title(L"(x,y)=(24,19), (r,\phi)=(\sqrt{13},%$(round(2+atan(-2/3)/pi,digits=2)) \pi)")
    # push!(images, axs[2,3].plot(θ,G4[21-3,21+2,:]))
    # axs[2,3].set_ylim(0,14)
    # axs[2,3].set_title(L"(x,y)=(18,23), (r,\phi)=(\sqrt{13},%$(round(2 + atan(2/3)/pi-1,digits=2)) \pi)")
    # push!(images, axs[2,4].plot(θ,G4[21-3,21-2,:]))
    # axs[2,4].set_ylim(0,14)
    # axs[2,4].set_title(L"(x,y)=(18,19), (r,\phi)=(\sqrt{13},%$(round(1-atan(2/3)/pi,digits=2)) \pi)")
    
    # fig.suptitle(L"D_T=%$(DT),D_R=%$(DR),v_0=%$(v0),\gamma=%$(γ)", fontsize=16)
    fig.suptitle(L"D_T=%$(DT),D_R=%$(DR),v_0=%$(v0),N_x=%$(Nx),N_\theta=%$(Nθ),\gamma=%$(γ)", fontsize=14)
    # push!(images,axs.scatter(x=rs2[1:end-1],y=[rho4[21,21]+(v0/DT)*sum(prs3[1:i])*(0.5*sqrt(2)/(npoints+1)) for i∈1:npoints]))
    push!(images,axs.scatter(x=rs2,y=[rho4[21,21]+(v0/DT)*int_p_r(rs2[i]) for i∈1:npoints]))
    push!(images,axs.scatter(x=[sqrt((x̃-0.5)^2+(ỹ-0.5)^2) for x̃∈x,ỹ∈y],y=[rho4[i,j] for i∈1:Nx,j∈1:Ny]))
    # push!(images,axs.scatter(x=[sqrt((x̃-0.5)^2+(ỹ-0.5)^2) for x̃∈x,ỹ∈y],y=[p_rs[i,j] for i∈1:Nx,j∈1:Ny]))
    # push!(images,axs.scatter(x=IJs,y=[rho4[21,21]+(v0/DT)*int_p_r(rs2[i]) for i∈1:npoints]))
    # push!(images,axs.scatter(x=[sqrt((x̃-0.5)^2+(ỹ-0.5)^2) for x̃∈x,ỹ∈y],y=[sqrt(P4[1,i,j]^2+P4[2,i,j]^2)/rho4[i,j] for i∈1:Nx,j∈1:Ny]))
    # push!(images,axs.scatter(x=[sqrt((x̃-0.5)^2+(ỹ-0.5)^2) for x̃∈x,ỹ∈y],y=[(RotMatrix{2}(-atan(y[j]-0.5,x[i]-0.5))*(P4[:,i,j]/rho4[i,j]))[2] for i∈1:Nx,j∈1:Ny]))
    # push!(images,axs.scatter(x=[sqrt((x̃-0.5)^2+(ỹ-0.5)^2) for x̃∈x,ỹ∈y],y=[C4[i,j] for i∈1:Nx,j∈1:Ny]))
    # push!(images,axs.scatter(x=[sqrt((x̃-0.5)^2+(ỹ-0.5)^2) for x̃∈x,ỹ∈y],y=[C4[21,21]+0.85*log.(rho4[i,j]/rho4[21,21]) for i∈1:Nx,j∈1:Ny]))
    # push!(images,axs.scatter(x=[sqrt((x̃-0.5)^2+(ỹ-0.5)^2) for x̃∈x,ỹ∈y],y=[C4[31,31]+log.(rho4[i,j]/rho4[30,30]) for i∈1:Nx,j∈1:Ny]))
    # push!(images,axs.scatter(x=[sqrt((x̃-0.5)^2+(ỹ-0.5)^2) for x̃∈x,ỹ∈y],y=[C4[i,j] for i∈1:Nx,j∈1:Ny]))
    # push!(images,axs.scatter(x=[sqrt((x̃-0.5)^2+(ỹ-0.5)^2) for x̃∈x,ỹ∈y],y=[log.(log.(rho4[i,j])) for i∈1:Nx,j∈1:Ny]))
    # axs.set_ylabel(L"\log\log\rho(x,y)")
    # axs.set_yscale("log")
    # axs.set_xlim(0,0.25)
    axs.set_xlim(-0.01,0.5)
    axs.set_xlabel(L"(x-\frac{1}{2})^2+(y-\frac{1}{2})^2",fontsize=14)
    # axs.legend([L"c(0)+0.85\log(\rho(r)/\rho(0))",L"c(r)"], fontsize = 17)

    axs.legend([L"\rho(0)+\frac{v_0}{D_T}\int_0^r\mathbf{p}_r(s)ds",L"\rho(r)"], fontsize = 17)
    # axs.legend([L"\mathbf{p}_r"], fontsize = 17)
    # push!(images, axs[1,2].plot(θ2,circshift(G4[21+2,21+3,:],16-Int(ceil(16*atan(3/2)/pi)))))
    # axs[1,1].set_xlabel(L"\theta-\phi")
    # axs[1,1].set_ylabel(L"f(x,y,\theta)")
    # axs[1,2].set_title(L"(x,y)=(23,24), (r,\phi)=(\sqrt{13},%$(round(atan(3/2)/pi,digits=2)) \pi)")
    # axs[1,2].set_ylim(0,14)
    # push!(images, axs[2,3].plot(θ2,circshift(G4[21+2,21-3,:],16-Int(ceil(16*(2*pi+atan(-3/2))/pi)))))
    # axs[2,3].set_ylim(0,14)
    # axs[2,3].set_title(L"(x,y)=(23,18), (r,\phi)=(\sqrt{13},%$(round(2+atan(-3/2)/pi,digits=2)) \pi)")
    # push!(images, axs[2,4].plot(θ2,circshift(G4[21-2,21-3,:],16-Int(ceil(16*(pi + atan(3/2))/pi)))))
    # axs[2,4].set_ylim(0,14)
    # axs[2,4].set_title(L"(x,y)=(19,18), (r,\phi)=(\sqrt{13},%$(round(2 + atan(3/2)/pi - 1,digits=2)) \pi)")
    # push!(images, axs[1,3].plot(θ2,circshift(G4[21-2,21+3,:],16-Int(ceil(16*(pi-atan(3/2))/pi)))))
    # axs[1,3].set_ylim(0,14)
    # axs[1,3].set_title(L"(x,y)=(19,24), (r,\phi)=(\sqrt{13},%$(round(1 - atan(3/2)/pi,digits=2)) \pi)")
    # push!(images, axs[1,1].plot(θ2,circshift(G4[21+3,21+2,:],16-Int(ceil(16*atan(2/3)/pi)))))
    # axs[1,1].set_ylim(0,14)
    # axs[1,1].set_title(L"(x,y)=(24,23), (r,\phi)=(\sqrt{13},%$(round(atan(2/3)/pi,digits=2)) \pi)")
    # push!(images, axs[2,2].plot(θ2,circshift(G4[21+3,21-2,:],16-Int(ceil(16*(2*pi+atan(-2/3))/pi)))))
    # axs[2,2].set_ylim(0,14)
    # axs[2,2].set_title(L"(x,y)=(24,19), (r,\phi)=(\sqrt{13},%$(round(2+atan(-2/3)/pi,digits=2)) \pi)")
    # push!(images, axs[1,4].plot(θ2,circshift(G4[21-3,21+2,:],16-Int(ceil(16*(pi - atan(2/3))/pi)))))
    # axs[1,4].set_ylim(0,14)
    # axs[1,4].set_title(L"(x,y)=(18,23), (r,\phi)=(\sqrt{13},%$(round(1-atan(2/3)/pi,digits=2)) \pi)")
    # push!(images, axs[2,1].plot(θ2,circshift(G4[21-3,21-2,:],16-Int(ceil(16*(pi + atan(2/3))/pi)))))
    # axs[2,1].set_ylim(0,14)
    # axs[2,1].set_title(L"(x,y)=(18,19), (r,\phi)=(\sqrt{13},%$(round(2 + atan(2/3)/pi-1,digits=2)) \pi)")
    
    # push!(images, axs.plot(θ2,circshift(G4[21+5,21+6,:],16-Int(round(16*atan(6/5)/pi)))))
    # fig.suptitle(L"D_T=%$(DT),D_R=%$(DR),v_0=%$(v0),N_x=%$(Nx),N_\theta=%$(Nθ),\gamma=%$(γ), r=2\Delta x", fontsize=14)

    # axs.set_xlabel(L"\theta-\phi")
    # axs.set_ylabel(L"\logf(x,y,\theta)")
    # axs.set_ylim(0.5,3.0)
    # push!(images, axs.vlines(x=0,ymin=0,ymax=3,linestyle="dashed",color="k"))
    # push!(images, axs.plot(θ2,log.(circshift(G4[21+1,21+1,:],16-Int(round(16*atan(1/1)/pi))))))
    # push!(images, axs.plot(θ2,log.(circshift(G4[21+1,21-1,:],16-Int(round(16*(2*pi+atan(-1/1))/pi))))))
    # push!(images, axs.plot(θ2,log.(circshift(G4[21-1,21-1,:],16-Int(round(16*(pi + atan(1/1))/pi))))))
    # push!(images, axs.plot(θ2,log.(circshift(G4[21-1,21+1,:],16-Int(round(16*(pi-atan(1/1))/pi))))))
    # push!(images, axs.plot(θ2,log.(circshift(G4[31+2,31+1,:],21-Int(round(21*atan(1/2)/pi))))))
    # push!(images, axs.plot(θ2,log.(circshift(G4[31+2,31-1,:],21-Int(round(21*(2*pi+atan(-1/2))/pi))))))
    # push!(images, axs.plot(θ2,log.(circshift(G4[31-2,31+1,:],21-Int(round(21*(pi - atan(1/2))/pi))))))
    # push!(images, axs.plot(θ2,log.(circshift(G4[31-2,31-1,:],21-Int(round(21*(pi + atan(1/2))/pi))))))

    # push!(images, axs.vlines(x=0,ymin=0,ymax=13,linestyle="dashed",color="k"))
    # fig.suptitle(L"D_T=%$(DT),D_R=%$(DR),v_0=%$(v0),N_x=%$(Nx),N_\theta=%$(Nθ),\gamma=%$(γ), centre \ (x,y)=(31,31), r=\sqrt{5}\Delta x", fontsize=12)
    # push!(images, axs.plot(θ2,log.(circshift(G4[31+3,31+1,:],16-Int(round(16*atan(1/3)/pi))))))
    # push!(images, axs.plot(θ2,log.(circshift(G4[31+3,31-1,:],16-Int(round(16*(2*pi+atan(-1/3))/pi))))))
    # push!(images, axs.plot(θ2,log.(circshift(G4[31-3,31-1,:],16-Int(round(16*(pi + atan(1/3))/pi))))))
    # push!(images, axs.plot(θ2,log.(circshift(G4[31-3,31+1,:],16-Int(round(16*(pi-atan(1/3))/pi))))))
    # push!(images, axs.plot(θ2,log.(circshift(G4[31+1,31+3,:],16-Int(round(16*atan(3/1)/pi))))))
    # push!(images, axs.plot(θ2,log.(circshift(G4[31+1,31-3,:],16-Int(round(16*(2*pi+atan(-3/1))/pi))))))
    # push!(images, axs.plot(θ2,log.(circshift(G4[31-1,31+3,:],16-Int(round(16*(pi - atan(3/1))/pi))))))
    # push!(images, axs.plot(θ2,log.(circshift(G4[31-1,31-3,:],16-Int(round(16*(pi + atan(3/1))/pi))))))
    # axs.set_xlabel(L"\theta-\phi")
    # axs.set_ylabel(L"\log f(x,y,\theta)")
    # axs.set_ylim(0,12)

    # push!(images, axs.plot(θ2,log.(circshift(G4[31+3,31+1,:],16-Int(round(16*atan(1/3)/pi))))))
    # push!(images, axs.plot(θ2,log.(circshift(G4[31+3,31-1,:],16-Int(round(16*(2*pi+atan(-1/3))/pi))))))
    # push!(images, axs.plot(θ2,log.(circshift(G4[31-3,31-1,:],16-Int(round(16*(pi + atan(1/3))/pi))))))
    # push!(images, axs.plot(θ2,log.(circshift(G4[31-3,31+1,:],16-Int(round(16*(pi-atan(1/3))/pi))))))
    # push!(images, axs.plot(θ2,log.(circshift(G4[31+1,31+3,:],16-Int(round(16*atan(3/1)/pi))))))
    # push!(images, axs.plot(θ2,log.(circshift(G4[31+1,31-3,:],16-Int(round(16*(2*pi+atan(-3/1))/pi))))))
    # push!(images, axs.plot(θ2,log.(circshift(G4[31-1,31+3,:],16-Int(round(16*(pi - atan(3/1))/pi))))))
    # push!(images, axs.plot(θ2,log.(circshift(G4[31-1,31-3,:],16-Int(round(16*(pi + atan(3/1))/pi))))))

    # push!(images,axs[3,1].plot(x,rho4[21,:]))
    # axs[3,1].set_xlabel(L"y")
    # axs[3,1].set_ylabel(L"\rho(x,y)")
    # axs[3,1].set_title(L"(x)=(21)")
    # push!(images,axs[3,2].plot(x,rho4[:,21]))
    # axs[3,2].set_xlabel(L"x")
    # axs[3,2].set_ylabel(L"\rho(x,y)")
    # axs[3,2].set_title(L"(y)=(21)")
    # push!(images,axs[3,3].plot(x,C4[:,21]))
    # # axs[3,3].set_yscale("log")
    # axs[3,3].set_xlabel(L"x")
    # axs[3,3].set_ylabel(L"c(x,y)")
    # axs[3,3].set_title(L"C \ at \ (y)=(21)")
    # # push!(images,axs[3,4].plot(x,[exp(-((x̃-0.5)^2)/(0.01)) for x̃ ∈ x]))
    # # axs[3,4].set_xlabel(L"x")
    # # axs[3,4].set_title(L"\propto\exp(-(x-0.5)^2)")
    # push!(images,axs[3,4].plot(x,[exp(exp(-((x̃-0.5)^2)/(0.01)))-1 for x̃ ∈ x]))
    # axs[3,4].set_xlabel(L"x")
    # axs[3,4].set_title(L"\propto\exp(\exp(-(x-0.5)^2))-1")
    # push!(images,axs[4,1].plot(x,Pn[1,21,:]))
    # axs[4,1].set_xlabel(L"y")
    # axs[4,1].set_ylabel(L"q(x,y)")
    # axs[4,1].set_title(L"(x)=(21)")
    # push!(images,axs[4,2].plot(x,Pn[1,:,21]))
    # axs[4,2].set_xlabel(L"x")
    # axs[4,2].set_ylabel(L"q(x,y)")
    # axs[4,2].set_title(L"(y)=(21)")
    # push!(images,axs[4,3].plot(x,[-abs((x̃-0.5))/(0.01)+50 for x̃ ∈ x]))
    # axs[4,3].set_xlabel(L"x")
    # axs[4,3].set_title(L"\propto -|x-0.5|+const")
    # push!(images,axs[4,4].plot(x,[abs((x̃-0.5))*exp(-((x̃-0.5)^2)/(0.05)) for x̃ ∈ x]))
    # axs[4,4].set_xlabel(L"x")
    # axs[4,4].set_title(L"\propto |x-0.5|\exp(-|x-0.5|^2)")

    # push!(images, axs[1,1].plot(θ,G4[18,21,:]))
    # axs[1,1].set_xlabel(L"\theta")
    # axs[1,1].set_ylabel(L"f(x,\theta)")
    # axs[1,1].set_title(L"D_T=%$(DT),D_R=%$(DR),v_0=%$(v0),\gamma=%$(γ)")
    # push!(images, axs[1,2].plot(θ,G4[18+1,21,:]))
    # push!(images, axs[2,1].plot(θ,G4[18,21-1,:]))
    # push!(images, axs[2,2].plot(θ,G4[18+1,21-1,:]))
    

    
    

    # fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm1, cmap = colmap), ax = axs[1,Nc])
    # fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm2, cmap = colmap), ax = axs[2,Nc])

    # axs[1,1].text(-0.3,.8,L"\rho",fontsize=16)
    # axs[2,1].text(-0.3,.8,L"{\bf p}/\rho",fontsize=16)

    display(fig)
    # fig.savefig(sn_eps)
    close(fig)

end