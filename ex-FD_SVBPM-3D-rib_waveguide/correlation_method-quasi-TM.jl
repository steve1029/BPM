include("../lib-FD-SVBPM-3D-PML.jl")

using .FD_SVBPM_2D
using Serialization
using Plots

function main()

    fname = "ex-FD_SVBPM-3D-rib_waveguide/"
    working_dir = joinpath(pwd(), fname)
    cd(working_dir)
    @show working_dir
    # working_dir = pwd()

    # um = 10^-6 # Never run Simulation with too-small values.
    # nm = 10^-9

    um = 1.
    nm = 10^-3

    Nx = 201
    Ny = 211
    Nz = 221

    Nxc = Int(round(Nx/2))
    Nyc = Int(round(Ny/2))
    Nzc = Int(round(Nz/2))

    @assert Nx % 2 == 1 # To include x=0.
    @assert Ny % 2 == 1 # To include y=0.

    Lx = 4*um
    Ly = 20*um
    Lz = 68*um

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

    w1 = 3*um
    w2 = 15*um
    h = 0.2*um
    d = 0.8*um
    n1, n2, n3 = 1., 3.44, 3.36
    n = get_rib_waveguide_profile(  x, y, z,
                                    w1, w2, h, d,
                                    n1, n2, n3;
                                    )

    nmax = maximum(real.(n))
    nmin = minimum(real.(n))
    
    nxy = heatmap(  y./um,
                    x./um,
                    real.(n[:,:,Nzc]);
                    dpi=300,
                    clim=(nmin, nmax),
                    colormap=:tab20,
                    xlabel="y (μm)",
                    ylabel="x (μm)",
                    zlabel="index",
                    title="Refractive index")

    nxz = heatmap(  z./um, 
                    x./um, 
                    real.(n[:,Nyc,:]);
                    dpi=300, 
                    clim=(nmin, nmax), 
                    colormap=:tab20,
                    xlabel="z (μm)", 
                    ylabel="x (μm)", 
                    zlabel="index", 
                    title="Refractive index")

    nyz = heatmap(  z./um, 
                    y./um, 
                    real.(n[Nxc,:,:]);
                    dpi=300, 
                    clim=(nmin, nmax), 
                    colormap=:tab20,
                    xlabel="z (μm)", 
                    ylabel="y (μm)", 
                    zlabel="index", 
                    title="Refractive index")

    savefig(nxy, working_dir*"refractive_index_profile-xy.png")
    savefig(nxz, working_dir*"refractive_index_profile-xz.png")
    savefig(nyz, working_dir*"refractive_index_profile-yz.png")

    nref = n3 
    λ = 1550*nm
    k0 = 2*π / λ
    β = k0 * nref

    Δnmax = nmax - nmin
    criterion = (λ/2 / nmin / Δnmax)
    @assert dz < criterion "dz must be less than $(criterion/um) um." # For details, refer to eq 2.106.

    wx = 0.5*um
    wy = 0.5*um
    xshift = 0.4*um
    yshift = 0.0*um
    Eplane, input_plot = get_gaussian_input(x, y, xshift, yshift, wx, wy; plot=true)
 
    α = 0.50001

    pol = "quasi-TM"
    nametag = "Efield_$pol"

    #=
    =#
    Efield = get_Efield(x, y, z, nref, n, λ, α, Eplane; pmlx=true, pmly=true, pol=pol)
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
    xloc004 = 0.4
    xloc010 = 1.0
    yloc000 = 0.0
    yloc040 = 4.0
    zloc080 = 8.0
    zloc200 = 20.0

    xzplot_y000 = plot_field(x, y, z, Efield, "xz", yloc000*um, clim=(0,Inf))
    xzplot_y040 = plot_field(x, y, z, Efield, "xz", yloc040*um, clim=(0,Inf))
    xyplot_z080 = plot_field(x, y, z, Efield, "xy", zloc080*um, clim=(0,Inf))
    xyplot_z200 = plot_field(x, y, z, Efield, "xy", zloc200*um, clim=(0,Inf))
    yzplot_x000 = plot_field(x, y, z, Efield, "yz", xloc000*um, clim=(0,Inf))
    yzplot_x004 = plot_field(x, y, z, Efield, "yz", xloc004*um, clim=(0,Inf))
    yzplot_x010 = plot_field(x, y, z, Efield, "yz", xloc010*um, clim=(0,Inf))

    savefig(xzplot_y000, working_dir*"$nametag-xz-$yloc000-profile.png")
    savefig(xzplot_y040, working_dir*"$nametag-xz-$yloc040-profile.png")
    savefig(xyplot_z080, working_dir*"$nametag-xy-$zloc080-profile.png")
    savefig(xyplot_z200, working_dir*"$nametag-xy-$zloc200-profile.png")
    savefig(yzplot_x000, working_dir*"$nametag-yz-$xloc000-profile.png")
    savefig(yzplot_x004, working_dir*"$nametag-yz-$xloc004-profile.png")
    savefig(yzplot_x010, working_dir*"$nametag-yz-$xloc010-profile.png")

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