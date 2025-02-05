using Plots
using LinearAlgebra
using FFTW
using Peaks
using Profile
using Serialization
using Printf

function get_gaussian_input(
    x::AbstractVector, 
    xshift::Float64, 
    w::Float64)::Vector
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

function get_Efield(
    Nx::Int, 
    Nz::Int, 
    Lx::Float64, 
    Lz::Float64, 
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

    Tm, Tp, R = PML(Nx, Nz, npml, dx, n, λ)

    Efield = zeros(ComplexF64, Nx, Nz)
    Efield[:,1] = input

    for k in 2:Nz
        Efield[:,k] = to_next(k, α, λ, dx, dz, n0, n, Tm, Tp, R, Efield[:,k-1])
    end

    return Efield
end

"""
function correlation_method

    Note that Pf is not a function of ξ.
    Note that since fftfreq returns frequency, we need 2*π to make angular freq.
"""
function correlation_method(Efield::AbstractMatrix, dx::Float64, dz::Float64)

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

function plot_withlayout(
    x::AbstractVector, 
    z::AbstractVector, 
    field::AbstractMatrix, 
    n0::Float64,
    Δn::Float64,
    n::Matrix{ComplexF64}, 
    input::AbstractVector, 
    corr::AbstractVector, 
    ξ::AbstractVector, 
    ξv::AbstractVector, 
    ξvind::AbstractVector, 
    peakh::AbstractVector, 
    Pξ_abs::AbstractVector, 
    figname::String; ymax::Number=1)

    um = 10^-6
    nm = 10^-9

    intensity = abs2.(field)
    anno = [(xi, h, "ξv=$xi") for (xi, h) in zip(ξv, peakh)]
    input_abs = (abs.(input).^2) 
    input = plot(x/um, input_abs/maximum(input_abs), 
                            label="input beam", 
                            xlabel="Normalized Intensity",
                            ylabel="x (μm)",
                            lw=1.5, dpi=300,)
    hm1 = heatmap(z, x/um, intensity, 
                    dpi=300, clim=(0,1), c=:thermal, 
                    xlabel="z (μm)", ylabel="x (μm)", zlabel="Intensity", 
                    title="Straight waveguide")
    hm2 = heatmap(z, x/um, real(n), 
                    dpi=300, 
                    clim=(n0, n0+Δn), 
                    xlabel="z (μm)", ylabel="x (μm)", zlabel="index", 
                    title="Refractive index")
    normed_corr = real.(corr) / maximum(real.(corr))
    Pz = plot(z, normed_corr,
                    xlabel="z (μm)", 
                    ylabel="x (μm)",
                    title="Correlation function")
            # Pfz = plot(ξ*um, modeintensity, xlim=(-0.2, 0.2), yscale=:log10)
    normed_Pξ_abs = Pξ_abs / maximum(Pξ_abs)
    Pfz_anal = plotpeaks(ξ*um, normed_Pξ_abs; 
                            peaks=ξvind, 
                            prominences=true, widths=true, 
                            xlim=(-0.2, 0.2),
                            # ylim=(0, ymax),
                            yscale=:log10,  
                            # ylim=(10^-34, -10^-1),
                            annotations=anno)

    plots = [input, hm1, hm2, Pz, Pfz_anal]
    # layout = @layout [a b c; d e{1,2}]
    # layout = grid(2, 3; widths=[1, 1, 2], heights=[0.5, 0.5])
 
    layout = @layout [grid(1,3); b{0.333w} c{0.666w}]
    plot(plots..., layout=layout, size=(1400, 1000))
    savefig(figname)
end

function get_h(
    Lx::Float64,
    Lz::Float64,
    α::Float64,
    mode_num::Int,
    Efield::Matrix{ComplexF64},
    n0::Float64,
    Δn::Float64,
    n::Matrix{ComplexF64},
    λ::Float64,
    ξv::Vector{Float64};
    ymax::Number=1
    )::Vector

    um = 10^-6
    nm = 10^-9

    Nx = size(Efield, 1)
    Nz = size(Efield, 2)

    x = range(-Lx/2, Lx/2; length=Nx)
    z = range(0, Lz; length=Nz)

    dx = step(x)
    dz = step(z)

    # k0 = 2*π / λ
    # # β = n0 * k0
    # βs = ξv .+ β

    # Since we inverted the sign of ξv, 
    # as FFTW returns -ξ, not ξ, 
    # roll back the sign of ξv.
    ξv = -ξv

    hfields = []
    hxys = []

    for v in 1:mode_num

        nametag = "h$(v-1)"
        figname = "./$nametag-after-propagation.png"

        # Get h0.
        if v == 1
            hfield = Efield
            μ=mode_num

        # Get the remaining h
        elseif v > 1 && v < mode_num
            hfield = hfields[v-1]
            μ=mode_num

        else
            hfield = hfields[v-1]
            μ=1

        end

        phasor = exp.(-1im*ξv[μ]*z) # since ξv is negetive here, minus sign is added.
        phasor_mat = repeat(transpose(phasor), Nx, 1)
        psi = hfield .* dz .* phasor_mat

        Δβ = ξv[μ]-ξv[v]
        lim = 2*π/ abs(Δβ)
        ind = argmin(abs.(z .- lim))
        @printf("μ=%d, v=%d, ξv=%f, intgration limit=%f um\n", μ, v, -ξv[v]*um, lim/um)
        h = vec(sum(psi[:,1:ind], dims=2))
        # h = h / maximum(abs.(h))

        push!(hxys, h)
        println("h$(v-1) calculated.")
        if v == mode_num
            return hxys
        end

        hfield = get_Efield(Nx, Nz, Lx, Lz, n0, n, λ, α, h)
        push!(hfields, hfield)

        corr_h, ξ_h, ξvind_h, ξv_h, peakh_h, Pξ_abs_h = correlation_method(hfield, dx, dz)
        plot_withlayout(x, z, hfield, n0, Δn, n, h, corr_h, ξ_h, ξv_h, ξvind_h, peakh_h, Pξ_abs_h, figname, ymax=ymax)
    end

    return hxys

end

function plot_h(
    x::AbstractVector,
    Eline::AbstractVector,
    hxys::AbstractVecOrMat{T})::Int where T

    insert!(hxys, 1, Eline)
    num = length(hxys)

    eachmode = Vector{T}(undef, num-1)
    eachmode[num-1] = hxys[num-1]

    for i in num-2:-1:1
        eachmode[i] = hxys[i] - hxys[i+1]
    end

    # labels = ["input"; ["mode $(i-1)" for i in 1:num]]
    labels = ["mode $(i-1)" for i in 1:(num-1)]

    plots = []

    um = 10^-6
    for (num, mode) in enumerate(eachmode)
        y = real.(mode) / maximum(abs.(real.(mode)))
        push!(plots, plot(x/um, y, 
                            dpi=300, 
                            label=labels[num], 
                            xlabel="x (μm)", 
                            ylabel="Normalized E field",
                            ylim=(-1, 1)
                            ))
    end

    n = length(plots)
    allplot = plot(plots..., dpi=300, layout=grid(n,1), size=(500, 200*n), link=:x)

    savefig(allplot, "./all_hxy.png")

    return 0
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

    @show dx / um
    @show dz / um

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

    Efield = get_Efield(Nx, Nz, Lx, Lz, n0, n, λ, α, Eline)

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
    plot_withlayout(x, z, Efield, n0, Δn, n, Eline, Pz, ξ, ξv, ξvind, peakh, Pξ_abs, figname; ymax=ymax)

    mode_num = 3
    Efield = deserialize("./Efield.dat")
    ξv = deserialize("./xiv_Efield.dat")
    ymax = deserialize("./correlation_function_abs_max.dat")
    hxys = get_h(Lx, Lz, α, mode_num, Efield, n0, Δn, n , λ, ξv; ymax=ymax)
    # hxys = get_h(Lx, Lz, α, mode_num, Efield, n0, Δn, n , λ, ξv)
    serialize("./hs.dat", hxys)

    hxys = deserialize("./hs.dat")
    plot_h(x, Eline, hxys)
    println("Simulation finished.")
end

@time main()