include("../lib-FD-SVBPM-TE-waveguide-PML.jl")

using .FD_SBPM_2D
using Serialization

function main()

    fname = "ex-FD_SBPM-2D_PML-asymmetric_step_index_waveguide/"
    working_dir = joinpath(pwd(), fname)
    cd(working_dir)
    @show working_dir

    # um = 10^-6 # Never run Simulation with too-small values.
    # nm = 10^-9

    um = 1
    nm = 10^-3

    Nx = 201
    Nz = 20000

    @assert Nx % 2 == 1 # To include x=0.

    Lx = 5*um
    Lz = 4000*um

    # dx = 0.05*um
    # dz = 0.2*um

    # Lx = (Nx-1)*dx
    # Lz = (Nz-1)*dz

    x = range(-Lx/2, Lx/2; length=Nx)
    z = range(0, Lz; length=Nz)

    dx = step(x)
    dz = step(z)

    @show dx / um
    @show dz / um

    n0 = 1.45
    loc_discont = [-2.5*um, -0.3*um, 0.3*um, 2.5*um]
    refractiveindice = [1.45, 1.95, 1]
    n = get_step_index_profile(x, z,
            loc_discont,
            refractiveindice;
            save=true,
            savedir=working_dir
            )

    λ = 1550*nm
    k0 = 2*π / λ
    β = k0 * n0
    Δnmax = maximum(real(n)) - n0
    @assert dz < (λ/2 / n0 / Δnmax / 5) # For details, refer to eq 2.106.
    w = 0.5*um
    xshift = 0*um
    Eline = get_gaussian_input(x, xshift, w)
 
    α = 0.50001

    ξdiff = 0.12
    nt = ξdiff/k0 + n0

    nametag = "Efield"
    Pzname = "Pz_$nametag.dat"
    ξvname = "xiv_$nametag.dat"
    ξvindname = "xiv_index_$nametag.dat"
    ξname = "xi_$nametag.dat"
    ymaxname = "correlation_function_abs_max.dat"
    peakhname = "peakh_$nametag.dat"
    Pξ_absname = "Pxi_abs_$nametag.dat"

    Efield = get_Efield(x, z, nt, n, λ, α, Eline)

    serialize(working_dir*"x.dat", x)
    serialize(working_dir*"z.dat", z)
    serialize(working_dir*"Efield.dat", Efield)

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
    Efield = deserialize(working_dir*"Efield.dat")
    Pz = deserialize(Pzname)
    ξ = deserialize(ξname)
    ξv = deserialize(ξvname)
    ξvind = deserialize(ξvindname)
    peakh = deserialize(peakhname)
    Pξ_abs = deserialize(Pξ_absname)
    ymax = deserialize(ymaxname)

    figname = "FD_SVBPM-2D-TE-waveguide-PML.png"
    plot_with_corr(x, z, Efield, n0, Δnmax, n, Eline, 
                    Pz, ξ, ξv, ξvind, peakh, Pξ_abs, 
                    figname; ymax=ymax, savedir=working_dir)

    # exit()
    mode_num = 1
    mode_transverse_profiles = get_h(Lx, Lz, α, mode_num, 
                                        Efield, nt, n0, Δnmax, n, 
                                        λ, ξv; ymax=ymax)
    # mode_profiles = get_h(Lx, Lz, α, mode_num, Efield, n0, Δnmax, n , λ, ξv)
    modename = working_dir*"mode_transverse_profiles.dat"
    serialize(modename, mode_transverse_profiles)

    mode_transverse_profiles = deserialize(modename)
    plot_mode(x, mode_transverse_profiles, ξv, λ, n0)
    println("Simulation finished.")
end

@time main()