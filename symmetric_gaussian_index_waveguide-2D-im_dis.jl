include("./FD-SBPM-2D-waveguide-PML.jl")

using Serialization
using .FD_SBPM_2D

function main()

    savedir = joinpath(pwd(),"im-dis-FD_SBPM_2D-PML/")

    if !isdir(savedir)
        mkdir(savedir)
        println("Folder created.")
    end

    um = 10^-6
    nm = 10^-9

    Nx = 400
    Nz = 200

    Lx = 20*um
    Lz = 50*um

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

    Efield = get_Efield(Nx, Nz, Lx, Lz, n0, n, λ, α, Eline)
    trialξ = k0 * (n0 + Δn*0.6)
    trialξ = k0 * n0 + 111/um
    @show trialξ*um
    # trialξ = 0
    psiτ = im_dis(Efield, z, β, trialξ)

    serialize(savedir*"x.dat", x)
    serialize(savedir*"z.dat", z)
    serialize(savedir*"Efield.dat", Efield)

    figname = "im_dis-Efield.png"
    plots = plot_field(x, z, psiτ, n0, Δn, n, Eline, figname; savedir=savedir, save=true)

    #=
    nametag = "Efield"
    Pz, ξ, ξvind, ξv, peakh, Pξ_abs= correlation_method(Efield, dx, dz)

    @show ξv*um

    serialize(savedir*"xiv_$nametag.dat", ξv)
    serialize(savedir*"xi_$nametag.dat", ξ)

    figname = savedir*"FD_SBPM-2D-waveguide-PML.png"
    ymax = maximum(Pξ_abs)*1.05
    serialize(savedir*"correlation_function_abs_max.dat", ymax)
    plot_withlayout(x, z, Efield, n0, Δn, n, Eline, Pz, ξ, ξv, ξvind, peakh, Pξ_abs, figname; ymax=ymax)

    mode_num = 3
    Efield = deserialize(savedir*"Efield.dat")
    ξv = deserialize(savedir*"xiv_Efield.dat")
    ymax = deserialize(savedir*"correlation_function_abs_max.dat")
    mode_transverse_profiles = get_h(Lx, Lz, α, mode_num, Efield, n0, Δn, n , λ, ξv; ymax=ymax)
    # mode_profiles = get_h(Lx, Lz, α, mode_num, Efield, n0, Δn, n , λ, ξv)
    serialize(savedir*"mode_transverse_profiles.dat", mode_transverse_profiles)

    ξv = deserialize(savedir*"xiv_Efield.dat")
    mode_transverse_profiles = deserialize(savedir*"mode_transverse_profiles.dat")
    plot_mode(x, mode_transverse_profiles, ξv, β)
    =#
    println("Simulation finished.")
end

@time main()