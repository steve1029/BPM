include("../lib-2D-FD-SBPM-waveguide-PML.jl")

using .FD_SBPM_2D
using Serialization
using Printf

function main()

    fname = "./ex-2D-FD-SBPM-PML-gaussian_index_waveguide/im-dis/"
    working_dir = joinpath(pwd(), fname)
    cd(working_dir)
    @show working_dir

    um = 1.
    nm = 10^-3

    Nx = 401
    Nz = 101

    Lx = 20*um
    Lz = 250*um

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
    @assert abs.(dz) < (λ/2 / n0 / Δn / 5)

    w = 2*um
    xshift = 1*um
    mode_to_get = 0

    if mode_to_get == 0
        Eline = get_gaussian_input(x, xshift, w)
    else
        Eline = deserialize(working_dir*"newinput_no_mode_$(mode_to_get-1).dat")
    end

    ntrial = 1.465016
    α = 0.5001

    @show ntrial

    newinput = 
        get_mode_profiles_im_dis(x, τ, 
                                    Eline, 
                                    n, 
                                    ntrial, 
                                    λ, 
                                    α;
                                    iternum=1,
                                    mode=mode_to_get,
                                    savedir=working_dir
                                )
    serialize(working_dir*"newinput_no_mode_$mode_to_get.dat", newinput)
    serialize(working_dir*"x.dat", x)
    serialize(working_dir*"z.dat", z)

    # filename = @sprintf("im_dis-neff-%.3f-FD_SBPM_2D-noPML.png", neff)
    # plot_field(x, z, Efield, n0, Δn, n, Eline, 
                # filename; savedir=savedir)

    println("Simulation finished.")
end

@time main()