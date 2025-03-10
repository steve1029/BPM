include("../lib-FD-SBPM-2D-waveguide-PML.jl")

using .FD_SBPM_2D
using Serialization

function main()

    fname = "ex-FD_SBPM-2D_PML-gaussian_index_waveguide/"
    working_dir = joinpath(pwd(), fname)
    cd(working_dir)
    @show working_dir

    um = 10^-6
    nm = 10^-9

    Nx = 400
    Nz = 10000

    Lx = 20*um
    Lz = 25000*um

    x = range(-Lx/2, Lx/2; length=Nx)
    z = range(0, Lz; length=Nz)

    dx = step(x)
    dz = step(z)

    @show dx / um
    @show dz / um

    n0 = 1.45
    σ = 2*um
    Δn = 0.02
    n = get_symmetric_Gaussian_index_profile(x, n0, σ, Δn, Nx, Nz)

    λ = 850*nm
    k0 = 2*π / λ
    β = k0 * n0
    @assert dz < (λ/2 / n0 / Δn)
    w = 2*um
    xshift = 1*um
    Eline = get_gaussian_input(x, xshift, w)
 
    α = 0.5001

    target = 0.12
    nt = target/k0 + n0

    nametag = "Efield"
    Pzname = "Pz_$nametag.dat"
    ξvname = "xiv_$nametag.dat"
    ξvindname = "xiv_index_$nametag.dat"
    ξname = "xi_$nametag.dat"
    ymaxname = "./correlation_function_abs_max.dat"
    peakhname = "peakh_$nametag.dat"
    Pξ_absname = "Pxi_abs_$nametag.dat"

    #=
    Efield = get_Efield(x, z, nt, n, λ, α, Eline)

    serialize("x.dat", x)
    serialize("z.dat", z)
    serialize("Efield.dat", Efield)

    Pz, ξ, ξvind, ξv, peakh, Pξ_abs= correlation_method(Efield, dx, dz)

    ymax = maximum(Pξ_abs)*1.05

    @show ξv*um
    @show peakh
    @show ξvind

    serialize(Pzname, Pz)
    serialize(ξvname, ξv)
    serialize(ξvindname, ξvind)
    serialize(ξname, ξ)
    serialize(ymaxname, ymax)
    serialize(peakhname, peakh)
    serialize(Pξ_absname, Pξ_abs)
    =#

    x = deserialize("./x.dat")
    z = deserialize("./z.dat")
    Efield = deserialize("./Efield.dat")
    Pz = deserialize(Pzname)
    ξ = deserialize(ξname)
    ξv = deserialize(ξvname)
    ξvind = deserialize(ξvindname)
    peakh = deserialize(peakhname)
    Pξ_abs = deserialize(Pξ_absname)
    ymax = deserialize(ymaxname)

    figname = "./FD_SBPM-2D-waveguide-PML.png"
    plot_with_corr(x, z, Efield, n0, Δn, n, Eline, 
                    Pz, ξ, ξv, ξvind, peakh, Pξ_abs, figname; ymax=ymax)

    mode_num = 3
    mode_transverse_profiles = get_h(Lx, Lz, α, mode_num, 
                                        Efield, nt, n0, Δn, n, 
                                        λ, ξv; ymax=ymax)
    # mode_profiles = get_h(Lx, Lz, α, mode_num, Efield, n0, Δn, n , λ, ξv)
    serialize("./mode_transverse_profiles.dat", mode_transverse_profiles)

    mode_transverse_profiles = deserialize("./mode_transverse_profiles.dat")
    plot_mode(x, mode_transverse_profiles, ξv, λ, n0)
    println("Simulation finished.")
end

@time main()