include("../lib-2D-FD-SVBPM-waveguide-PML-TE-TM.jl")

using .FD_SVBPM_2D
using Serialization
using Plots

function main()

    fname = "ex-FD_SVBPM-2D_TE_TM_PML-asymmetric_step_index_waveguide/"
    working_dir = joinpath(pwd(), fname)
    # cd(working_dir)
    @show working_dir
    # working_dir = pwd()

    # um = 10^-6 # Never run Simulation with too-small values.
    # nm = 10^-9

    um = 10
    nm = 10^-2

    Nx = 501
    Nz = 10001

    @assert Nx % 2 == 1 # To include x=0.

    Lx = 5*um
    Lz = 1000*um

    # dx = 0.05*um
    # dz = 0.2*um

    # Lx = (Nx-1)*dx
    # Lz = (Nz-1)*dz

    x = range(-Lx/2, Lx/2; length=Nx)
    z = range(0, Lz; length=Nz)
    τ = 1im.*z

    dx = step(x)
    dz = step(z)
    dτ = step(τ)

    @show dx / um
    @show dz / um

    loc_discont = [-2.5*um, -0.3*um, 0.3*um, 2.5*um]
    refractiveindice = [1.45, 1.95, 1]
    n = get_step_index_profile(x, z,
                                loc_discont,
                                refractiveindice;
                                save=true,
                                savedir=working_dir
                                )

    # ξdiff = 0.12
    # nref = ξdiff/k0 + n0
    # nref = n0
    # @show nref

    nref = 1.45
    λ = 1550*nm
    k0 = 2*π / λ
    β = k0 * nref
    nmin = minimum(real(n))
    nmax = maximum(real(n))
    Δnmax = nmax - nmin
    criterion = (λ/2 / nmin / Δnmax / 5)
    @assert dz < criterion "dz must be less than $(criterion/um) um." # For details, refer to eq 2.106.
    w = 0.5*um
    xshift = 0*um
 
    α = 0.50001

    pol = "TM"
    mode_to_get = 1

    if mode_to_get == 0
        Eline = get_gaussian_input(x, xshift, w)
    else
        Eline = deserialize(savedir*"newinput_no_mode_$(mode_to_get-1).dat")
    end

    newinput = 
        get_mode_profiles_im_dis(x, τ, Eline, n, ntrial, λ, α, 4; 
                                    mode=mode_to_get,
                                    savedir=savedir
                                    )
    serialize(savedir*"newinput_no_mode_$mode_to_get.dat", newinput)
    serialize(savedir*"x.dat", x)
    serialize(savedir*"z.dat", z)


    #=
    Efield = get_Efield(x, z, nref, n, λ, α, Eline; pml=true, pol=pol)

    nametag = "Efield_$pol"
    Pzname = "$nametag-Pz.dat"
    ξvname = "$nametag-xiv.dat"
    ξvindname = "$nametag-xiv_index.dat"
    ξname = "$nametag-xi.dat"
    ymaxname = "$nametag-correlation_function_abs_max.dat"
    peakhname = "$nametag-peakh.dat"
    Pξ_absname = "$nametag-Pxi_abs.dat"

    serialize(working_dir*"x.dat", x)
    serialize(working_dir*"z.dat", z)
    serialize(working_dir*"$nametag.dat", Efield)

    inputplot, Iplot, nplot = plot_field(x, z, Efield, n, Eline)
    layout = @layout [grid(1,3)]
    allplots = plot([inputplot, Iplot, nplot]..., layout=layout, size=(1400, 500))
    savefig(allplots, working_dir*"$nametag-profile.png")

    Pz, ξ, ξvind, ξv, peakh, Pξ_abs= correlation_method(Efield, dx, dz)

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

    x = deserialize(working_dir*"x.dat")
    z = deserialize(working_dir*"z.dat")
    Efield = deserialize(working_dir*"$nametag.dat")
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