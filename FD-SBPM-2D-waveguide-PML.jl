using Plots
using LinearAlgebra
using FFTW
using Peaks
using Profile
using Serialization

function get_gaussian_input(x, xshift, w)::Vector
    Eline = exp.(-((x.-xshift).^2 ./ (w^2))) .+ 0im
    return Eline
end

function get_step_index_profile(
    Nx::Int, 
    Nz::Int,
    slabthick::Float64,
    cladthick::Float64,
    slabindex::Float64,
    cladindex::Float64
    )::Matrix
    
    n = ones(ComplexF64, Nx, Nz)
    bcm = Int(round(Nx/2 - slabthick/2/dx - cladthick/2/dx))
    bsm = Int(round(Nx/2 - slabthick/2/dx))
    bsp = Int(round(Nx/2 + slabthick/2/dx))
    bcp = Int(round(Nx/2 + slabthick/2/dx + cladthick/2/dx))
    n[bcm:bsm,:] .= cladindex
    n[bsm:bsp,:] .= slabindex 
    n[bsp:bcp,:] .= cladindex

    return n
end

function get_symmetric_Gaussian_index_profile(
    x,
    n0::Float64,
    σ::Float64,
    Δn::Float64,
    Nx::Int64, 
    Nz::Int64)::Matrix
    
    n = ones(ComplexF64, Nx, Nz)
    n .*= n0 .+ Δn .* repeat(exp.(-(x ./ σ).^2), 1, Nz) # broadcast a vector to a matrix.

    return n
end

"""
function to_next
    In this function, we used 'exp(i(wt-kr))' notation.

# Arguments
# Returns
# Example
"""
function to_next(
    step::Int,
    α::Float64,
    λ::Float64,
    dx::Float64, 
    dz::Float64, 
    n0::Float64,
    n::Matrix{ComplexF64}, 
    Tm::Matrix{ComplexF64}, 
    Tp::Matrix{ComplexF64}, 
    R::Matrix{ComplexF64}, 
    E::Vector{ComplexF64}
    )::Vector{ComplexF64}

    k0 = 2*π / λ
    nd = n[:,step].^2 .- n0^2

    b = (2*α.* R[:, step] /dx^2) .- (α.*nd.*k0^2) .+ (2im*k0*n0/dz)
    a = (-α/dx^2) .* Tm[:, step]
    c = (-α/dx^2) .* Tp[:, step]
    A = diagm(-1=>a, 0=>b, 1=>c)

    D = (1-α)*k0^2 .* nd .- (2*(1-α)/dx^2 .* R[:, step]) .+ (2im*k0*n0/dz)
    above = ((1-α) / dx^2) .* Tp[:, step]
    below = ((1-α) / dx^2) .* Tm[:, step]

    B = diagm(-1=>below, 0=>D, 1=>above)

    r = B * E
    newE = A \ r

    return newE
end

"""
function PML

    In this function, we used 'exp(i(wt-kr))' notation.

# Arguments
# Returns
# Example
"""
function PML(
    Nx::Int64,
    Nz::Int64,
    npml::Int64,
    dx::Float64,
    np::Matrix{ComplexF64}, 
    λ::Float64)::Tuple{Matrix{ComplexF64}, Matrix{ComplexF64}, Matrix{ComplexF64}}

    rc0 = 1.e-16
    μ0 = 4*π*10^-7
    ε0 = 8.8541878128e-12
    c = sqrt(1/μ0/ε0)
    imp = sqrt(μ0/ε0)
    ω = 2*π*c / λ

    go = 2.
    bdwx = (npml-1) * dx
    σmax = -(go+1) * log(rc0) / (2*imp*bdwx)
    # σmax = 5*ε0*ω

    loc = range(0, 1; length=npml)
    σx = σmax .* loc.^go

    q = ones(ComplexF64, Nx, Nz)
    ll = np[      npml:-1:  1,:]
    rr = np[end+1-npml: 1:end,:]

    q[      npml:-1:  1, :] = 1 ./ (1 .- ((1im.*σx) ./ (ω .* ε0 .* ll.^2)))
    q[end+1-npml: 1:end, :] = 1 ./ (1 .- ((1im.*σx) ./ (ω .* ε0 .* rr.^2)))

    Tm = q[2:end  ,:] .* (q[2:end  ,:] .+ q[1:end-1,:]) ./ 2
    Tp = q[1:end-1,:] .* (q[1:end-1,:] .+ q[2:end  ,:]) ./ 2

    R = ones(ComplexF64, Nx, Nz) 
    R .*= (q.^2 ./ 2)
    R[2:end  ,:] += q[2:end  ,:] .* q[1:end-1,:] ./ 4
    R[1:end-1,:] += q[1:end-1,:] .* q[2:end  ,:] ./ 4

    return Tm, Tp, R
end

"""
function get_correlation

    Note that Pf is not a function of ξ.
    Note that since fftfreq returns frequency, we need 2*π to make angular freq.
"""
function get_correlation(Efield, dx, dz, nametag)

    Nz = size(Efield, 2)
    Pz = zeros(ComplexF64, Nz)
    Pf = zeros(ComplexF64, Nz)

    con = conj(Efield[:,1])
    for k in axes(Efield, 2)
        elwise = con .* Efield[:,k] 
        Pz[k] = sum(elwise)*dx
    end

    Pf = fft(Pz) # Pf is not a function of \xi.
    F = fftshift(Pf)
    Pξ_abs = abs.(F)

    fs = 1/dz
    ξ = -fftshift(fftfreq(Nz, fs).*(2*π))

    pks = findmaxima(Pξ_abs, 20)
    ξv = ξ[pks.indices]

    ξvind = pks.indices
    peakh = pks.heights

    return Pz, ξ, ξvind, ξv, peakh, Pξ_abs 
end

function get_Efield(
    Nx::Int, 
    Nz::Int, 
    Lx, 
    Lz, 
    n0::Float64, 
    n::Matrix{ComplexF64}, 
    λ::Float64, 
    α::Float64, 
    input::Vector{ComplexF64}
    )::Matrix{ComplexF64}

    um = 10^-6
    nm = 10^-9

    npml = 10

    x = range(-Lx/2, Lx/2; length=Nx)
    z = range(0, Lz; length=Nz)

    dx = step(x)
    dz = step(z)

    @show dx / um
    @show dz / um

    Tm, Tp, R = PML(Nx, Nz, npml, dx, n, λ)

    Efield = zeros(ComplexF64, Nx, Nz)
    Efield[:,1] = input

    @time for k in 2:Nz
        Efield[:,k] = to_next(k, α, λ, dx, dz, n0, n, Tm, Tp, R, Efield[:,k-1])
    end

    return Efield
end

function plot_withlayout(x, z, field, n, input, corr, ξ, ξv, ξvind, peakh, Pξ_abs, figname)

    um = 10^-6
    nm = 10^-9

    intensity = abs2.(field)
    anno = [(xi, h, "ξv=$xi") for (xi, h) in zip(ξv, peakh)]
    input_beam_plot = plot(x, abs.(input).^2, label="input beam", lw=1.5, dpi=300, size=(500,500))
    hm1 = heatmap(z, x, intensity, dpi=300, clim=(0,1), c=:thermal, xlabel="z (μm)", ylabel="x (μm)", zlabel="Intensity", title="Straight waveguide")
    hm2 = heatmap(z, x, real(n), dpi=300, clim=(-Inf,Inf), xlabel="z (μm)", ylabel="x (μm)", zlabel="index", title="Refractive index")
    Pz = plot(z, real.(corr))
    # Pfz = plot(ξ*um, modeintensity, xlim=(-0.2, 0.2), yscale=:log10)
    Pfz_anal = plotpeaks(ξ*um, Pξ_abs; 
                            peaks=ξvind, prominences=true, widths=true, 
                            yscale=:log10, xlim=(-0.2, 0.2),
                            annotations=anno)

    @show ξvind
    @show ξv*um

    # plots = [input_beam_plot, hm1, hm2, Pz, Pfz]
    plots = [input_beam_plot, hm1, hm2, Pz, Pfz_anal]
    # layout = @layout [a b c; d e{1,2}]
    # layout = grid(2, 3; widths=[1, 1, 2], heights=[0.5, 0.5])
 
    layout = @layout [grid(1,3); b{0.333w} c{0.666w}]
    plot(plots..., layout=layout, size=(1400, 1000))
    savefig(figname)
end

function get_h(
    Lx,
    Lz,
    α,
    mode_num::Int,
    Efield::Matrix{ComplexF64},
    n0::Float64,
    n::Matrix{ComplexF64},
    λ::Float64,
    dz::Float64,
    ξv::Vector{Float64}
    )::Matrix{ComplexF64}

    Nx = size(Efield, 1)
    Nz = size(Efield, 2)

    x = range(-Lx/2, Lx/2; length=Nx)
    z = range(0, Lz; length=Nz)

    k0 = 2*π / λ
    β = n0 * k0
    βs = ξv .+ β

    hs = []

    @time for v in 1:mode_num

        # Get h0.
        if v == 1
            Δβ = βs[2]-βs[1]
            phasor = exp.(1im*Δβ*z)
            lim = 2*π/ Δβ
            ind = argmin(abs.(z .- lim))
            psi = Efield .* dz .* repeat(phasor, Nx, 1)
            h0 = vec(sum(psi[:,1:ind], dims=2))
            push!(hs, h0)

        # Get the remaining h
        else
            oldh = hs[v-1]
            hfield = get_Efield(Nx, Nz, Lx, Lz, n0, n, λ, α, oldh)

            nametag = "h$v"
            figname = "./$nametag-profile.png"

            corr, ξ, ξvind, ξv, peakh, Pξ_abs = get_correlation(field, dx, dz, nametag)
            plot_withlayout(x, z, hfield, n, oldh, corr, ξ, ξv, ξvind, peakh, Pξ_abs, figname)

            lim = 2*π/(ξμ-ξv[1])
            ind = argmin(abs.(z .- lim))
            newh = sum(hfield[:,1:ind])
            push!(hs, newh)
        end

    end

    return hs

end

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

    n0 = 1.45
    σ = 2*um
    Δn = 0.02
    n = get_symmetric_Gaussian_index_profile(x, n0, σ, Δn, Nx, Nz)

    λ = 850*nm
    @assert dz < (λ/2 / n0 / Δn)
    w = 2*um
    xshift = 1*um
    Eline = get_gaussian_input(x, xshift, w)
 
    α = 0.5001
    #=
    Efield = get_Efield(Nx, Nz, Lx, Lz, n0, n, λ, α, Eline)

    serialize("x.dat", x)
    serialize("z.dat", z)
    serialize("Efield.dat", Efield)

    nametag = "Efield"
    Pz, ξ, ξvind, ξv, peakh, Pξ_abs= get_correlation(Efield, dx, dz, nametag)

    serialize("xiv_$nametag.dat", ξv)
    serialize("xi_$nametag.dat", ξ)

    figname = "./FD_SBPM-2D-waveguide-PML.png"
    plot_withlayout(x, z, Efield, n, Eline, Pz, ξ, ξv, ξvind, peakh, Pξ_abs, figname)

    =#
    mode_num = 3
    Efield = deserialize("./Efield.dat")
    ξv = deserialize("./xiv_Efield.dat")
    hs = get_h(Lx, Lz, α, mode_num, Efield, n0, n , λ, dz, ξv)
end

main()