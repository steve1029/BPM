module FD_SBPM_2D

using Plots
using LinearAlgebra
using FFTW
using Peaks
using Profile
using Printf

# const um = 10^-6
# const nm = 10^-9

const um = 1.
const nm = 10^-3

export get_gaussian_input, 
        get_step_index_profile, 
        get_symmetric_Gaussian_index_profile,
        get_Efield, 
        correlation_method, 
        plot_field, 
        plot_with_corr, 
        get_h, 
        plot_mode, 
        get_mode_profiles_im_dis

function get_gaussian_input(
    x::AbstractVector, 
    xshift::Number, 
    w::Number)::Vector
    Eline = exp.(-((x.-xshift).^2 ./ (w^2))) .+ 0im
    return Eline
end

function get_step_index_profile(
    x::AbstractVector, 
    z::AbstractVector, 
    loc_discont::AbstractVector,
    refractiveindice::AbstractVector;
    save = false,
    savedir = "./"
    )::Matrix

    Nx = length(x)
    Nz = length(z)

    dx = step(x)

    n = ones(ComplexF64, Nx, Nz)

    @assert Nx % 2 == 1 "Nx must be odd!"
    loc_discont_ind = Int64.(round.(loc_discont ./ dx) .+ ((Nx+1)/2))

    @show loc_discont_ind
    for (i, loc) in enumerate(loc_discont_ind)

        if i == 1
            continue
        else 
            srt = loc_discont_ind[i-1]
            n[srt:loc, :] .= refractiveindice[i-1]
            @show refractiveindice[i-1]
        end
    end

    if save == true
        hm = heatmap(abs.(z)./um, x./um, real(n), 
                        dpi=300, 
                        # clim=(n0, n0+Δn), 
                        xlabel="z (μm)", ylabel="x (μm)", zlabel="index", 
                        title="Refractive index")

        savefig(savedir*"refractive_index_profile.png")
    end

    return n
end

function get_symmetric_Gaussian_index_profile(
    x,
    n0::Number,
    σ::Number,
    Δn::Number,
    Nx::Int64, 
    Nz::Int64)::Matrix
    
    n = ones(ComplexF64, Nx, Nz)
    n .*= n0 .+ Δn .* repeat(exp.(-(x ./ σ).^2), 1, Nz) # broadcast a vector to a matrix.

    return n
end

function get_Efield(
    x::AbstractVector,
    z::AbstractVector,
    nt::Number, 
    n::Matrix{ComplexF64}, 
    λ::Number, 
    α::Number, 
    input::Vector{ComplexF64};
    pml = true
    )::Matrix{ComplexF64}

    npml = 10

    Nx = length(x) 
    Nz = length(z)

    dx = step(x)
    dz = step(z)

    Efield = zeros(ComplexF64, Nx, Nz)
    Efield[:,1] = input

    if pml == false
        Tm = ones(eltype(Efield), Nx-1, Nz)
        Tp = ones(eltype(Efield), Nx-1, Nz)
        R  = ones(eltype(Efield), Nx, Nz)
    else
        Tm, Tp, R = _pml(Nx, Nz, npml, dx, n, λ)
    end

    for step in 2:Nz
        Efield[:,step] = _to_next(step, α, λ, 
                                    dx, dz, 
                                    nt, n, 
                                    Tm, Tp, R, 
                                    Efield[:,step-1];
                                    )
    end

    return Efield
end

"""
function _pml

    In this function, we used 'exp(i(wt-kr))' notation.

# Arguments
# Returns
# Example
"""
function _pml(
    Nx::Int64,
    Nz::Int64,
    npml::Int64,
    dx::Number,
    np::Matrix{ComplexF64}, 
    λ::Number)::Tuple{Matrix{ComplexF64}, Matrix{ComplexF64}, Matrix{ComplexF64}}

    rc0 = 1.e-16
    μ0 = 4*π*10^-7
    ε0 = 8.8541878128e-12
    c = sqrt(1/μ0/ε0)
    imp = sqrt(μ0/ε0)
    ω = 2*π*c / λ

    go = 2. # grading order.
    bdwx = (npml-1) * dx
    σmax = -(go+1) * log(rc0) / (2*imp*bdwx)
    # σmax = 5*ε0*ω

    loc = range(0, 1; length=npml)
    σx = σmax .* (loc.^go)

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
function _to_next
    In this function, we used 'exp(i(wt-kr))' notation.

# Arguments
# Returns
# Example
"""
function _to_next(
    step::Int,
    α::Number,
    λ::Number,
    dx::Number, 
    dz::Number, 
    nt::Number,
    n::Matrix{ComplexF64}, 
    Tm::Matrix{ComplexF64}, 
    Tp::Matrix{ComplexF64}, 
    R::Matrix{ComplexF64}, 
    E::Vector{ComplexF64};
    )::Vector{ComplexF64}

    k0 = 2*π / λ
    nd = n[:,step].^2 .- nt^2

    b = (2*α.* R[:, step] ./ dx^2) .- (α.*nd.*k0^2) .+ (2im*k0*nt/dz)
    a = (-α/dx^2) .* Tm[:, step]
    c = (-α/dx^2) .* Tp[:, step]
    A = diagm(-1=>a, 0=>b, 1=>c)

    D = (1-α)*k0^2 .* nd .- (2*(1-α)/dx^2 .* R[:, step]) .+ (2im*k0*nt/dz)
    above = ((1-α) / dx^2) .* Tp[:, step]
    below = ((1-α) / dx^2) .* Tm[:, step]

    B = diagm(-1=>below, 0=>D, 1=>above)

    r = B * E
    newE = A \ r

    return newE
end


"""
function correlation_method

    Note that Pf is not a function of ξ.
    Note that since fftfreq returns frequency, 
    we need 2*π to make angular freq.
"""
function correlation_method(
    Efield::AbstractMatrix, 
    dx::Number, 
    dz::Number
    )

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
    # Since FFT returns with inverted sign,
    # there should be minus sign in front of ξ.
    ξ = -fftshift(fftfreq(Nz, fs).*(2*π)) 

    pks = findmaxima(Pξ_abs, 20)

    ξvind = pks.indices
    ξv = ξ[ξvind]
    peakh = pks.heights

    return Pz, ξ, ξvind, ξv, peakh, Pξ_abs 
end

function plot_field(
    x::AbstractVector, 
    z::AbstractVector, 
    field::AbstractMatrix, 
    n0::Number,
    Δn::Number,
    n::AbstractMatrix{<:Number}, 
    input::AbstractVector,
    figname::String;
    savedir="./", 
    save=true)
    
    intensity = abs2.(field)
    input_abs = (abs.(input).^2) 
    inputplot = plot(x./um, input_abs./maximum(input_abs), 
                            label="input beam", 
                            xlabel="Normalized Intensity",
                            ylabel="x (μm)",
                            lw=1.5, dpi=300,)
    hm1 = heatmap(abs.(z)./um, x./um, intensity, 
                    dpi=300, clim=(0,1), c=:thermal, 
                    xlabel="z (μm)", ylabel="x (μm)", zlabel="Intensity", 
                    title="Straight waveguide")
    hm2 = heatmap(abs.(z)./um, x./um, real(n), 
                    dpi=300, 
                    # clim=(n0, n0+Δn), 
                    xlabel="z (μm)", ylabel="x (μm)", zlabel="index", 
                    color=:blues,
                    title="Refractive index")

    plots = [inputplot, hm1, hm2]
    layout = @layout [grid(1,3)]
    plot(plots..., layout=layout, size=(1400, 500))

    if save == true
        savefig(savedir*figname)
    end

    return plots
end

function plot_with_corr(
    x::AbstractVector,
    z::AbstractVector, 
    field::AbstractMatrix, 
    n0::Number,
    Δn::Number,
    n::Matrix{ComplexF64}, 
    input::AbstractVector, 
    Pz::AbstractVector, 
    ξ::AbstractVector, 
    ξv::AbstractVector, 
    ξvind::AbstractVector, 
    peakh::AbstractVector, 
    Pξ_abs::AbstractVector, 
    figname::String; 
    ymax::Number=Inf, 
    savedir="./")

    profileplots = plot_field(x, z, field, n0, Δn, n, input, figname; 
                                savedir=savedir, save=false)
    
    normed_Pξ_abs = Pξ_abs ./ maximum(Pξ_abs)
    normed_Pz = real.(Pz) ./ maximum(real.(Pz))
    # anno = [(xi, h, "ξv=$xi") for (xi, h) in zip(ξv, peakh)]
    Pz_plot = plot(z, normed_Pz, 
                    dpi=300,
                    xlabel="z (μm)",
                    ylabel="normed_Re(Pz)",
                    title="Correlation function",
                    label="P(z)",
                    # ylim=(0,1)
                    )
    Pfz_anal = plotpeaks(ξ*um, normed_Pξ_abs; 
                            peaks=ξvind, 
                            prominences=true, widths=true, 
                            # xlim=(-0.2, 0.2),
                            # ylim=(0,ymax),
                            yscale=:log10,  
                            # ylim=(10^-34, -10^-1),
                            # annotations=anno)
                        )

    # @show typeof(profileplots)
    # @show typeof(Pfz_anal)
    # @show typeof([corr, Pfz_anal])
    # layout = @layout [a b c; d e{1,2}]
    # layout = grid(2, 3; widths=[1, 1, 2], heights=[0.5, 0.5])

    push!(profileplots, Pz_plot)
    push!(profileplots, Pfz_anal)
    # @show typeof(profileplots)

    layout = @layout [grid(1,3); b{0.333w} c{0.666w}]
    plot(profileplots..., layout=layout, size=(1400, 1000))
    savefig(savedir*figname)
end

function get_h(
    Lx::Number,
    Lz::Number,
    α::Number,
    mode_num::Int,
    Efield::Matrix{ComplexF64},
    nt::Number,
    n0::Number,
    Δn::Number,
    n::Matrix{ComplexF64},
    λ::Number,
    ξv::AbstractVector;
    ymax::Number=Inf
    )::AbstractVector

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

    k0 = 2*π/λ
    β = n0*k0
    neff = (β.+ξv)./k0

    mode_transverse_profiles = []
    # h = Vector{eltype(Efield)}(0+0im, length(x))
    # h = fill(zero(eltype(Efield)), length(x))
    h = Efield[:,end]

    # Get the μ-th mode.
    for μ in 1:mode_num

        psifields = [Efield]
        iter_num = 1
        nametag = "h$(μ-1)"

        for v in 1:mode_num

            if v == μ 
                # We are trying to get μ-th mode.
                # If v == μ, then μ-th mode will be eliminated from the field!
                # So, this case should be neglected!
                continue

            elseif v != μ 

                psi = psifields[iter_num]

                # since ξv is negetive here, minus sign is added.
                phasor = exp.(-1im*ξv[μ]*z) 
                phasor_mat = repeat(transpose(phasor), Nx, 1)
                integrand = psi .* dz .* phasor_mat

                Δβ = ξv[μ]-ξv[v]
                lim = 2*π / abs(Δβ)
                ind = argmin(abs.(z .- lim))
                # get rid of mode v from integrand.
                h = vec(sum(integrand[:,1:ind], dims=2)) 
                # h = h / maximum(abs.(h))

                nametag = nametag * "$(v-1)"
                @printf("mode %1d subtracted from E field to get \
                            mode %1d, \
                            neff=%9.6f, \
                            ξv=%9.6f, \
                            integration limit=%7.3f um. \
                            Got %s.\n", 
                            (v-1), 
                            (μ-1), 
                            neff[μ],
                            -ξv[μ]*um, 
                            lim/um, 
                            nametag
                            )
                # println("h$(v-1) calculated.")
                # if iter_num == (mode_num-1)
                # end

                psi = get_Efield(x, z, nt, n, λ, α, h)
                push!(psifields, psi)

                figname = "./$nametag-after-propagation.png"
                corr_h, ξ_h, ξvind_h, ξv_h, peakh_h, Pξ_abs_h = 
                    correlation_method(psi, dx, dz)
                plot_with_corr(x, z, psi, n0, Δn, n, h, corr_h, 
                                ξ_h, ξv_h, ξvind_h, peakh_h, Pξ_abs_h, 
                                figname; ymax=ymax)

                iter_num += 1
            end
        end
        push!(mode_transverse_profiles, h)
        println("$nametag has been added to mode profiles.")
    end

    return mode_transverse_profiles

end

function plot_mode(
    x::AbstractVector,
    mode_profiles::AbstractVecOrMat{T},
    ξv::AbstractVector,
    λ::Number,
    n0::Number
    )::Int where T

    num = length(mode_profiles)

    # labels = ["input"; ["mode $(i-1)" for i in 1:num]]

    k0 = 2*π / λ
    β = k0 * n0
    labels = ["mode $(i-1), β$i=$(@sprintf("%.7f", (ξv[i]+β)/k0))" for i in 1:num]

    plots = []

    for (num, mode) in enumerate(mode_profiles)
        y = real.(mode) / maximum(abs.(real.(mode)))
        push!(plots, plot(x/um, y, 
                            dpi=300, 
                            label=labels[num], 
                            legend=:bottomright,
                            legendfontsize=6,
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

function get_mode_profiles_im_dis(
    x::AbstractVector,
    τ::AbstractVector,
    uline::AbstractVector,
    n::AbstractMatrix, 
    ntrial::Number,
    λ::Number,
    α::Number;
    iternum::Number=2,
    mode = 0,
    figsave = true,
    savedir = "./"
    )

    dx = step(x)
    dτ = step(τ)
    k = 2*π / λ
    weightedβ = ntrial*k
    newinput = uline

    # Get mode 0.
    for i in 1:iternum
        ufield = get_Efield(x, τ, 
                            ntrial, n, λ, α, 
                            newinput; 
                            pml=false
                            )

        weightedξv = real(( log(sum(ufield[:,end  ])*dx) -
                            log(sum(ufield[:,end-1])*dx))*1im/dτ)
        weightedβ += weightedξv

        zlast = real(τ[end]*(-1im))
        af = ufield[:,end] ./ exp(weightedξv*zlast)
        # af = ufield[:,end]

        # @show zlast/um
        # @show weightedξv
        # @show weightedξv*zlast
        # @show exp(weightedξv*zlast)

        figname = "get_mode_$mode-trial_$i.png"

        plot_im_dis(x, τ, newinput, 
                    ufield, i, af, 
                    ntrial, 
                    figname; 
                    savedir=savedir,
                    figsave=figsave)

        st = @sprintf( "Trial %2d: \
                        ntrial=%9.6f, \
                        weightedξv/k=%9.6f, \
                        redefine n as:%9.6f\n",
                        i,
                        ntrial, 
                        weightedξv/k, 
                        weightedβ/k
                        )
        print(st)

        ntrial = weightedβ/k
        newinput .-= af
    end

    return newinput
end # function end.

function plot_im_dis(
    x, 
    τ, 
    uline, 
    ufield,
    iternum::Number,
    af,
    ntrial,
    figname;
    figsave=true,
    savedir="./",
    )

    intensity = abs2.(ufield)
    input_abs = (abs.(uline).^2) 

    # normed_input_abs = input_abs./maximum(input_abs)
    inplot = plot(x./um, input_abs, 
                            label="input beam", 
                            title="Input",
                            xlabel="z (μm)",
                            ylabel="x (μm)",
                            lw=1.5, dpi=300,)

    irplot = heatmap(abs.(τ)./um, x./um, intensity, 
                    dpi=300, 
                    clim=(0,Inf), 
                    c=:thermal, 
                    xlabel="z (μm)", 
                    ylabel="x (μm)", 
                    zlabel="Intensity", 
                    title="Straight waveguide")

    nst = @sprintf("ntrial=%8.6f", ntrial)
    afplot = plot(x./um, real.(af), 
                            label="f(x,∞)", 
                            title="\$a_$(iternum-1)f_$(iternum-1)(x,y), \$"*nst,
                            xlabel="τ",
                            ylabel="x (μm)",
                            lw=1.5, 
                            dpi=300
                            )
    # annotate!(fplot, 5, -0.2, text(nst, 12, :blue))
    plots = [inplot, irplot, afplot]
    layout = @layout [grid(1,3)]
    plot(plots..., 
            layout=layout, 
            size=(1400, 500)
            )

    if figsave == true
        savefig(savedir*figname)
    end

end # function end.

end # module end.