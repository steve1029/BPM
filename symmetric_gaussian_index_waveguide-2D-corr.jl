include("FD-SBPM-2D-waveguide-PML.jl")

using .FD_SBPM_2D

function main()

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
    ntrial = 1.46
    β = k0 * ntrial
    @assert dz < (λ/2 / n0 / Δn)
    w = 2*um
    xshift = 1*um
    Eline = get_gaussian_input(x, xshift, w)
 
    α = 0.5001

    Efield = get_Efield(x, z, n0, n, λ, α, Eline)

    serialize("x.dat", x)
    serialize("z.dat", z)
    serialize("Efield.dat", Efield)

    nametag = "Efield"
    Pz, ξ, ξvind, ξv, peakh, Pξ_abs= correlation_method(Efield, dx, dz)

    @show ξv*um

    serialize("xiv_$nametag.dat", ξv)
    serialize("xi_$nametag.dat", ξ)

    figname = "./FD_SBPM-2D-waveguide-PML.png"
    ymax = maximum(Pξ_abs)*1.05
    serialize("./correlation_function_abs_max.dat", ymax)
    plot_with_corr(x, z, Efield, n0, Δn, n, Eline, Pz, ξ, ξv, ξvind, peakh, Pξ_abs, figname; ymax=ymax)

    mode_num = 3
    Efield = deserialize("./Efield.dat")
    ξv = deserialize("./xiv_Efield.dat")
    ymax = deserialize("./correlation_function_abs_max.dat")
    mode_transverse_profiles = get_h(Lx, Lz, α, mode_num, Efield, n0, Δn, n , λ, ξv; ymax=ymax)
    # mode_profiles = get_h(Lx, Lz, α, mode_num, Efield, n0, Δn, n , λ, ξv)
    serialize("./mode_transverse_profiles.dat", mode_transverse_profiles)

    ξv = deserialize("./xiv_Efield.dat")
    mode_transverse_profiles = deserialize("./mode_transverse_profiles.dat")
    plot_mode(x, mode_transverse_profiles, ξv, β)
    println("Simulation finished.")
end

@time main()