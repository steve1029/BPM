include("../lib-2D-FD-SBPM-waveguide-PML-final.jl")

using .FD_SBPM_2D
using Serialization
using Printf

function main()

    #=
    savedir = joinpath(pwd(),"./im-dis-FD_SBPM_2D-PML/")

    if !isdir(savedir)
        mkdir(savedir)
        println("Folder created.")
    end
    =#

    savedir = joinpath(pwd(), "ex-FD_SBPM-2D_PML-gaussian_index_waveguide/im-dis/")
    @show savedir
    # um = 10^-6
    # nm = 10^-9

    um = 1.
    nm = 10^-3

    Nx = 101
    Nz = 101

    Lx = 20*um
    Lz = 120*um

    x = range(-Lx/2, Lx/2; length=Nx)
    z = range(0, Lz; length=Nz)
    τ = 1im.*z

    dx = step(x)
    dz = step(z)
    dτ = step(τ)

    @show dx / um
    @show dz / um

    n0 = 1.45
    σ = 2*um
    Δn = 0.02
    n = get_symmetric_Gaussian_index_profile(x, n0, σ, Δn, Nx, Nz)

    λ = 850*nm
    k0 = 2*π / λ
    ntrial = 1.465016
    Δntrial = ntrial - n0 
    Δβ = Δntrial*k0
    α = 0.5001

    @show ntrial
    @assert abs.(dz) < (λ/2 / n0 / Δn / 5)

    w = 2*um
    xshift = 1*um
    mode_to_get = 1

    if mode_to_get == 0
        Eline = get_gaussian_input(x, xshift, w)
    else
        Eline = deserialize(savedir*"newinput_no_mode_$(mode_to_get-1).dat")
    end

    newinput = 
        get_mode_profiles_im_dis(x, τ, Eline, n, ntrial, λ, α;
                                    iternum=5,
                                    mode=mode_to_get,
                                    savedir=savedir
                                    )
    serialize(savedir*"newinput_no_mode_$mode_to_get.dat", newinput)
    serialize(savedir*"x.dat", x)
    serialize(savedir*"z.dat", z)

    # filename = @sprintf("im_dis-neff-%.3f-FD_SBPM_2D-noPML.png", neff)
    # plot_field(x, z, Efield, n0, Δn, n, Eline, 
                # filename; savedir=savedir)

    println("Simulation finished.")
end

@time main()