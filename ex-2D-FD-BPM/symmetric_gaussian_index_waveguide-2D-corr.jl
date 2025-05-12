include("../lib-2D-FD-BPM-PML-E-Scalar-TE-TM-final.jl")

using .FD_BPM_2D
using Serialization
using Plots

function main()

    fname = "./ex-2D-FD-BPM/"
    working_dir = joinpath(pwd(), fname)
    cd(working_dir)
    @show working_dir

    um = 1
    nm = 10^-3

    Nx = 401
    Nz = 101

    Lx = 20*um
    Lz = 250*um

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

    # pol = "Scalar" # "TE" or "TM" or "Scalar"
    # pol = "TE" # "TE" or "TM" or "Scalar"
    pol = "TM" # "TE" or "TM" or "Scalar" 

    if pol == "TE"
        nametag = "Ey"
    elseif pol == "TM"
        nametag = "Ex"
    elseif pol == "Scalar"
        nametag = "E"
    end

    Pzname = "Pz_$nametag.dat"
    ξvname = "xiv_$nametag.dat"
    ξvindname = "xiv_index_$nametag.dat"
    ξname = "xi_$nametag.dat"
    ymaxname = "./correlation_function_abs_max.dat"
    peakhname = "peakh_$nametag.dat"
    Pξ_absname = "Pxi_abs_$nametag.dat"

    Efield = get_Efield(x, z, nt, n, λ, α, Eline; 
                        pml=true,
                        pol=pol,
                        )

    inputplot, Iplot, nplot = plot_field(x, z, Efield, n, Eline)
    layout = @layout [grid(1,3)]
    allplots = plot([inputplot, nplot, Iplot]..., layout=layout, size=(1400, 500))
    savefig(allplots, working_dir*"$pol-$nametag-p.png")

    serialize("$pol-$nametag-x.dat", x)
    serialize("$pol-$nametag-z.dat", z)
    serialize("$pol-$nametag-f.dat", Efield)

    #=
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
    =#
    println("Simulation finished.")
end

@time main()