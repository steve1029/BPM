include("../lib-2D-FD-BPM-PML-E-Scalar-TE-TM-final.jl")

using .FD_BPM_2D
using Serialization
using Plots

function main()

    fname = "./ex-2D-FD-BPM/"
    working_dir = joinpath(pwd(), fname)
    cd(working_dir)
    @show working_dir

    um = 1.
    nm = 10^-3

    Nx = 401
    Nz = 201

    Lx = 20*um
    Lz = 250*um

    x = range(-Lx/2, Lx/2; length=Nx)
    z = range(0, Lz; length=Nz)
    τ = 1im.*z

    serialize(working_dir*"x.dat", x)
    serialize(working_dir*"z.dat", z)

    dx = step(x)
    dz = step(z)
    dτ = step(τ)

    @show dx / um
    @show dz / um

    nclad = 1.45
    σ = 2*um
    Δn = 0.02
    n = get_symmetric_Gaussian_index_profile(x, nclad, σ, Δn, Nx, Nz)

    λ = 850*nm
    k0 = 2*π / λ
    α = 0.50001

    nmin = minimum(real(n))
    nmax = maximum(real(n))
    Δnmax = nmax - nmin
    criterion = (λ/2 / nmin / Δnmax / 5)
    @assert dz < criterion "dz must be less than $(criterion/um) um." # For details, refer to eq 2.106.

    w = 2*um
    xshift = 1*um
    Eline = get_gaussian_input(x, xshift, w)

    pol = "Scalar"
    if pol == "TE"
        nametag = "Ey"
    elseif pol == "TM"
        nametag = "Ex"
    elseif pol == "Scalar"
        nametag = "E"
    end

    pml = true
    Efield = get_Efield(x, z, 
                        nclad, n, λ, α, 
                        Eline, 
                        pol,
                        pml
                        )

    Pz, ξ, ξvind, ξv, av, Pξ_abs= correlation_method(Efield, dx, dz)

    figname = "./correlation_result.png"
    # Efield = deserialize("./Efield.dat")
    profileplots = plot_with_corr(x, z, Efield, n, Eline, 
                    Pz, ξ, ξv, ξvind, av, Pξ_abs)

    layout = @layout [grid(1,3); b{0.333w} c{0.666w}]
    allplots = plot(profileplots..., layout=layout, size=(1400, 1000))
    savefig(allplots, working_dir*figname)

    @show ξv*um
    @show av
    @show ξvind

    ymax = maximum(Pξ_abs)*1.05

    Pzname = "Pz_$nametag.dat"
    ξvname = "xiv_$nametag.dat"
    ξvindname = "xiv_index_$nametag.dat"
    ξname = "xi_$nametag.dat"
    ymaxname = "./correlation_function_abs_max.dat"
    avname = "av_$nametag.dat"
    Pξ_absname = "Pxi_abs_$nametag.dat"

    serialize(Pzname, Pz)
    serialize(ξvname, ξv)
    serialize(ξvindname, ξvind)
    serialize(ξname, ξ)
    serialize(ymaxname, ymax)
    serialize(avname, av)
    serialize(Pξ_absname, Pξ_abs)

    x = deserialize("./x.dat")
    z = deserialize("./z.dat")
    Pz = deserialize(Pzname)
    ξ = deserialize(ξname)
    ξv = deserialize(ξvname)
    ξvind = deserialize(ξvindname)
    av = deserialize(avname)
    Pξ_abs = deserialize(Pξ_absname)
    ymax = deserialize(ymaxname)

    mode_num = 3
    mode_transverse_profiles = get_h(Lx, Lz, α, mode_num, 
                                        Efield, nclad, n, 
                                        λ, ξv, pol, pml;
                                        savedir=working_dir,
                                        ymax=ymax)

    serialize("./mode_transverse_profiles.dat", mode_transverse_profiles)
    mode_transverse_profiles = deserialize("./mode_transverse_profiles.dat")
    plots = plot_mode(x, mode_transverse_profiles, ξv, λ, nclad)
 
    nmodes = length(plots)
    allplot = plot(plots..., dpi=300, 
                    layout=grid(nmodes,1), 
                    size=(500, 200*nmodes), link=:x)

    savefig(allplot, "./all_hxy.png")

    #=
    ### Imaginary Distance Method ###
    mode_to_get = 0
    if mode_to_get == 0
        Eline = get_gaussian_input(x, xshift, w)
    else
        Eline = deserialize(working_dir*"newinput_no_mode_$(mode_to_get-1).dat")
    end

    # Eline = deserialize(working_dir*"newinput_no_mode_0.dat")

    ntrial = 1.466
    iternum = 1

    newinput = 
        get_mode_profiles_im_dis(x, τ, 
                                    Eline, 
                                    pol,
                                    n, 
                                    ntrial, 
                                    λ, 
                                    α;
                                    iternum=iternum,
                                    modenum=mode_to_get,
                                    nametag=nametag,
                                    savedir=working_dir
                                )
    serialize(working_dir*"newinput_no_mode_$mode_to_get.dat", newinput)
    =#

    println("Simulation finished.")
end

@time main()