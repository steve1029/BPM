include("../lib-FD-SBPM-3D-PML-S_bend.jl")

import Printf as pf
using .FD_SVBPM_2D
using Serialization
using Plots

function main()

    fname = "ex-FD_SBPM-3D_PML-S_bend/"
    working_dir = joinpath(pwd(), fname)
    cd(working_dir)
    @show working_dir
    # working_dir = pwd()

    # um = 10^-6 # Never run Simulation with too-small values.
    # nm = 10^-9

    um = 1.
    nm = 10^-3

    Nx = 181
    Ny = 181
    Nz = 201

    Nxc = Int(round(Nx/2))
    Nyc = Int(round(Ny/2))
    Nzc = Int(round(Nz/2))

    @assert Nx % 2 == 1 # To include x=0.
    @assert Ny % 2 == 1 # To include y=0.

    Lx = 18*um
    Ly = 18*um
    Lz = 100*um

    # dx = 0.05*um
    # dz = 0.2*um

    # Lx = (Nx-1)*dx
    # Lz = (Nz-1)*dz

    x = range(-Lx/2, Lx/2; length=Nx)
    y = range(-Ly/2, Ly/2; length=Ny)
    z = range(0, Lz; length=Nz)

    dx = step(x)
    dy = step(y)
    dz = step(z)

    @show dx / um
    @show dy / um
    @show dz / um

    d = 3*um
    nc = 1.0
    ns = 1.45
    Δn = 0.025
    w = 1.0*um
    ywidth = 15*um
    n = get_S_bend_profile(  x, y, z,
                            nc, ns, Δn,
                            d, w, ywidth;
                        )

    nmax = maximum(real.(n))
    nmin = minimum(real.(n))
    
    nxy = heatmap(  y./um,
                    x./um,
                    real.(n[:,:,Nzc]);
                    dpi=300,
                    # clim=(nmin, nmax),
                    # colormap=:tab20,
                    # colormap=:cyclic_grey_15_85_c0_n256_s25,
                    xlabel="y (μm)",
                    ylabel="x (μm)",
                    zlabel="index",
                    title="Refractive index")

    nxz = heatmap(  z./um, 
                    x./um, 
                    real.(n[:,Nyc,:]);
                    dpi=300, 
                    # clim=(nmin, nmax), 
                    # colormap=:tab20,
                    # colormap=:cyclic_grey_15_85_c0_n256_s25,
                    xlabel="z (μm)", 
                    ylabel="x (μm)", 
                    zlabel="index", 
                    title="Refractive index")

    nyz = heatmap(  z./um, 
                    y./um, 
                    real.(n[Nxc-1,:,:]);
                    dpi=300, 
                    # clim=(nmin, nmax), 
                    # colormap=:tab20,
                    # colormap=:cyclic_grey_15_85_c0_n256_s25,
                    xlabel="z (μm)", 
                    ylabel="y (μm)", 
                    zlabel="index", 
                    title="Refractive index")

    savefig(nxy, working_dir*"refractive_index_profile-xy.png")
    savefig(nxz, working_dir*"refractive_index_profile-xz.png")
    savefig(nyz, working_dir*"refractive_index_profile-yz.png")

    nref = ns 
    λ = 1550*nm
    k0 = 2*π / λ
    β = k0 * nref

    Δnmax = nmax - nmin
    criterion = (λ/2 / nmin / Δnmax)
    @assert dz < criterion "dz must be less than $(criterion/um) um." # For details, refer to eq 2.106.

    wx = 0.5*um
    wy = 0.5*um
    xshift = -0.0*um
    yshift = 0.0*um
    Eplane, input_plot = get_gaussian_input(x, y, xshift, yshift, wx, wy; plot=true)
 
    α = 0.50001

    pol = "scalar"
    nametag = "Efield_$pol"

    #=
    =#
    Efield = get_Efield(x, y, z, nref, n, λ, α, Eplane; npml=10, pmlx=true, pmly=true)
    Pzname = "$nametag-Pz.dat"
    ξvname = "$nametag-xiv.dat"
    ξvindname = "$nametag-xiv_index.dat"
    ξname = "$nametag-xi.dat"
    ymaxname = "$nametag-correlation_function_abs_max.dat"
    peakhname = "$nametag-peakh.dat"
    Pξ_absname = "$nametag-Pxi_abs.dat"

    serialize(working_dir*"x.dat", x)
    serialize(working_dir*"y.dat", y)
    serialize(working_dir*"z.dat", z)
    serialize(working_dir*"$nametag.dat", Efield)

    x = deserialize(working_dir*"x.dat")
    y = deserialize(working_dir*"y.dat")
    z = deserialize(working_dir*"z.dat")
    Efield = deserialize(working_dir*"$nametag.dat")

    xloc000 = 0.0
    xloc004 = -0.4
    xloc010 = -1.0
    yloc000 = 0.0
    yloc040 = 4.0
    zloc080 = 8.0
    zloc200 = 20.0
    zloc300 = 30.0
    zloc500 = 50.0

    xloc000s = pf.@sprintf("%5.1f", xloc000)
    xloc004s = pf.@sprintf("%5.1f", xloc004)
    xloc010s = pf.@sprintf("%5.1f", xloc010)
    yloc000s = pf.@sprintf("%5.1f", yloc000)
    yloc040s = pf.@sprintf("%5.1f", yloc040)
    zloc080s = pf.@sprintf("%5.1f", zloc080)
    zloc200s = pf.@sprintf("%5.1f", zloc200)
    zloc300s = pf.@sprintf("%5.1f", zloc300)
    zloc500s = pf.@sprintf("%5.1f", zloc500)

    xzplot_y000 = plot_field(x, y, z, Efield, "xz", yloc000*um, clim=(0,Inf))
    xzplot_y040 = plot_field(x, y, z, Efield, "xz", yloc040*um, clim=(0,Inf))
    xyplot_z080 = plot_field(x, y, z, Efield, "xy", zloc080*um, clim=(0,Inf))
    xyplot_z200 = plot_field(x, y, z, Efield, "xy", zloc200*um, clim=(0,Inf))
    xyplot_z300 = plot_field(x, y, z, Efield, "xy", zloc300*um, clim=(0,Inf))
    xyplot_z500 = plot_field(x, y, z, Efield, "xy", zloc500*um, clim=(0,Inf))
    yzplot_x000 = plot_field(x, y, z, Efield, "yz", xloc000*um, clim=(0,Inf))
    yzplot_x004 = plot_field(x, y, z, Efield, "yz", xloc004*um, clim=(0,Inf))
    yzplot_x010 = plot_field(x, y, z, Efield, "yz", xloc010*um, clim=(0,Inf))

    savefig(xzplot_y000, working_dir*"$nametag-xz-($yloc000s)μm-profile.png")
    savefig(xzplot_y040, working_dir*"$nametag-xz-($yloc040s)μm-profile.png")
    savefig(xyplot_z080, working_dir*"$nametag-xy-($zloc080s)μm-profile.png")
    savefig(xyplot_z200, working_dir*"$nametag-xy-($zloc200s)μm-profile.png")
    savefig(xyplot_z300, working_dir*"$nametag-xy-($zloc300s)μm-profile.png")
    savefig(xyplot_z500, working_dir*"$nametag-xy-($zloc500s)μm-profile.png")
    savefig(yzplot_x000, working_dir*"$nametag-yz-($xloc000s)μm-profile.png")
    savefig(yzplot_x004, working_dir*"$nametag-yz-($xloc004s)μm-profile.png")
    savefig(yzplot_x010, working_dir*"$nametag-yz-($xloc010s)μm-profile.png")

    #=
    layout = @layout [grid(1,3)]
    allplots = plot([xzplot, xyplot, yzplot]..., layout=layout, size=(1400, 500))
    savefig(allplots, working_dir*"$nametag-profile.png")

    Pz, ξ, ξvind, ξv, peakh, Pξ_abs= correlation_method(Efield, dx, dz)

    n0 = 3.36
    ξdiff = 0.12
    nref = ξdiff/k0 + n0
    nref = n0
    @show nref
# 
    @show ξv
    neff = (β .+ ξv) ./ k0 
    @show neff

    ymax = maximum(Pξ_abs)*1.05

    serialize(Pzname, Pz)
    serialize(ξvname, ξv)
    serialize(ξvindname, ξvind)
    serialize(ξname, ξ)
    serialize(ymaxname, ymax)
    serialize(peakhname, peakh)
    serialize(Pξ_absname, Pξ_abs)

    Pz = deserialize(Pzname)
    ξ = deserialize(ξname)
    ξv = deserialize(ξvname)
    ξvind = deserialize(ξvindname)
    peakh = deserialize(peakhname)
    Pξ_abs = deserialize(Pξ_absname)
    ymax = deserialize(ymaxname)

    figname = "FD_SVBPM-2D-$pol-waveguide-PML.png"
    corrplots = plot_with_corr(x, z, Efield, n, Eline, 
                                Pz, ξ, ξv, ξvind, peakh, Pξ_abs
                                )

    layout = @layout [grid(1,3); b{0.333w} c{0.666w}]
    all_in_one = plot(corrplots..., layout=layout, size=(1400, 1000))
    savefig(all_in_one, working_dir*figname)

    # exit()
    mode_num = 1
    mode_transverse_profiles = get_h(Lx, Lz, α, 
                                        mode_num, 
                                        Efield, 
                                        nref, 
                                        n, 
                                        λ, 
                                        ξv; ymax=ymax)

    modename = working_dir*"$nametag-mode_transverse_profiles.dat"
    serialize(modename, mode_transverse_profiles)

    mode_transverse_profiles = deserialize(modename)
    modeplots = plot_mode(x, mode_transverse_profiles, ξv, λ, nref)

    nplots = length(modeplots)
    allplot = plot(modeplots..., 
                    dpi=300, 
                    layout=grid(nplots,1), 
                    size=(500, 200*nplots), 
                    link=:x)

    savefig(allplot, "./$pol-all_modes.png")
    =#
    println("Simulation finished.")
end

@time main()