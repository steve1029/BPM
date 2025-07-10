module FD_BPM_2D

using Plots
using Plots.PlotMeasures
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
        plot_analysis,
        plot_various_corr,
        plot_transverse_profile,
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

    # @show loc_discont_ind

    for (i, loc) in enumerate(loc_discont_ind)

        if i == 1
            continue
        else 
            srt = loc_discont_ind[i-1]
            n[srt:loc, :] .= refractiveindice[i-1]
            # @show refractiveindice[i-1]
        end
    end

    nmax = maximum(real(n))
    nmin = minimum(real(n))
    
    if save == true
        hm = heatmap(abs.(z)./um, x./um, real(n), 
                        dpi=300, 
                        clim=(nmin, nmax), 
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
    nref::Number, 
    n::Matrix{ComplexF64}, 
    λ::Number, 
    α::Number, 
    input::Vector{ComplexF64},
    pol::String,
    pml::Bool,
    )::Matrix{ComplexF64}

    npml = 10

    Nx = length(x) 
    Nz = length(z)

    dx = step(x)
    dz = step(z)

    Efield = zeros(ComplexF64, Nx, Nz)
    Efield[:,1] = input

    if pml == false
        # M = ones(eltype(Efield), Nx-1, Nz)
        # N = ones(eltype(Efield), Nx-1, Nz)
        # R = ones(eltype(Efield), Nx, Nz) .* 2
        # println("No PML applied.")
        M, N, R = _pml(Nx, Nz, npml, dx, n, λ; pol=pol, turnon=false)
    else
        M, N, R = _pml(Nx, Nz, npml, dx, n, λ; pol=pol, turnon=true)
    end

    for step in 2:Nz
        Efield[:,step] = _to_next_z(step, α, λ, 
                                        dx, dz, 
                                        nref, n, 
                                        M, N, R, 
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
    n::Matrix{ComplexF64}, 
    λ::Number;
    pol::String="TM",
    turnon::Bool=true
    )::Tuple{Matrix{ComplexF64}, Matrix{ComplexF64}, Matrix{ComplexF64}}

    @assert pol == "TE" || pol == "TM" || pol == "Scalar" "Invalid polarization type. Use 'TE', 'TM' or 'Scalar'."
    @assert turnon == true || turnon == false "Invalid PML option. Use 'true' or 'false'."

    rc0 = 1.e-16
    μ0 = 4*π*10^-7
    ε0 = 8.8541878128e-12
    c = sqrt(1/μ0/ε0)
    imp = sqrt(μ0/ε0)
    ω = 2*π*c / λ

    go = 2. # grading order.
    bdwx = (npml-1) * dx

    if turnon == true
        σmax = -(go+1) * log(rc0) / (2*imp*bdwx)
        # σmax = 5*ε0*ω
    else
        σmax = 0
    end

    loc = range(0, 1; length=npml)
    σx = σmax .* (loc.^go)

    ll = n[      npml:-1:  1,:]
    rr = n[end+1-npml: 1:end,:]

    q = ones(ComplexF64, Nx, Nz)
    q[      npml:-1:  1, :] = 1 ./ (1 .- ((1im.*σx) ./ (ω .* ε0 .* ll.^2)))
    q[end+1-npml: 1:end, :] = 1 ./ (1 .- ((1im.*σx) ./ (ω .* ε0 .* rr.^2)))

    qp = q[2:end  ,:]
    qm = q[1:end-1,:]

    np = n[2:end  ,:]
    nm = n[1:end-1,:]

    R = zeros(ComplexF64, Nx, Nz) 

    if pol == "TM"

        qn = (qp .+ qm)./(np.^2 .+ nm.^2)

        M = qp .* qn .* (nm.^2)
        N = qm .* qn .* (np.^2)

        R[2:end  ,:] .+= qp .* qn .* (np.^2)
        R[1:end-1,:] .+= qm .* qn .* (nm.^2)

    elseif pol == "TE" || pol == "Scalar"

        M = qp .* (qp .+ qm) ./ 2
        N = qm .* (qm .+ qp) ./ 2

        R .+= q.^2
        R[2:end  ,:] += qp .* qm ./ 2
        R[1:end-1,:] += qm .* qp ./ 2
    else
        error("Invalid polarization type. Use 'TE', 'TM' or 'Scalar'.")
    end

    return M, N, R
end


"""
function _to_next_z
    In this function, we used 'exp(i(wt-kr))' notation.

# Arguments
# Returns
# Example
"""
function _to_next_z(
    step::Int,
    α::Number,
    λ::Number,
    dx::Number, 
    dz::Number, 
    nref::Number,
    n::Matrix{ComplexF64}, 
    M::Matrix{ComplexF64}, 
    N::Matrix{ComplexF64}, 
    R::Matrix{ComplexF64}, 
    E::Vector{ComplexF64};
    )::Vector{ComplexF64}

    k0 = 2*π / λ
    nd = n[:,step].^2 .- nref^2

    b = (2im*k0*nref/dz) .+ (α.* R[:, step] ./ dx^2) .- (α.*nd.*k0^2)
    a = (-α/dx^2) .* M[:, step]
    c = (-α/dx^2) .* N[:, step]
    A = diagm(-1=>a, 0=>b, 1=>c)

    D = (2im*k0*nref/dz) .- ((1-α)/dx^2 .* R[:, step]) .+ ((1-α)*k0^2 .* nd)
    above = ((1-α) / dx^2) .* N[:, step]
    below = ((1-α) / dx^2) .* M[:, step]

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
    modal_weights = sqrt.(pks.heights)

    return Pz, ξ, ξvind, ξv, modal_weights, Pξ_abs 
end

function plot_field(
    x::AbstractVector, 
    z::AbstractVector, 
    field::AbstractMatrix, 
    n::Matrix{ComplexF64}, 
    input::AbstractVector)

    nmax = maximum(real(n))
    nmin = minimum(real(n))
    
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
                    clim=(nmin, nmax), 
                    xlabel="z (μm)", ylabel="x (μm)", zlabel="index", 
                    color=:blues,
                    title="Refractive index")

    return inputplot, hm1, hm2
end

function plot_transverse_profile(
    x::AbstractVector, 
    z::AbstractVector, 
    field::AbstractMatrix,
    zloc::Number
    )

    dz = step(z)

    zloc_int = Int64(round(zloc ./ dz))+1 # zloc is in μm, dz is in μm.

    tfield = field[:, zloc_int]
    # intensity = abs2.(tfield)
    real_tfield = real.(tfield)
    tfieldplot = plot(x./um, real_tfield./maximum(real_tfield), 
                            label="$(zloc) um",
                            title="Transverse profile at z=$(zloc) um", 
                            xlabel="z (μm)",
                            ylabel="x (μm)",
                            lw=1.5, dpi=300,)

    return tfieldplot

end

function plot_analysis(
    x::AbstractVector,
    z::AbstractVector, 
    field::AbstractMatrix, 
    n::Matrix{ComplexF64}, 
    input::AbstractVector, 
    Pz::AbstractVector, 
    ξ::AbstractVector, 
    ξv::AbstractVector, 
    ξvind::AbstractVector, 
    peakh::AbstractVector, 
    Pξ_abs::AbstractVector;)

    inputplot, Iplot, nplot = plot_field(x, z, field, n, input)

    profileplots = [inputplot, Iplot, nplot]
    
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

    return profileplots
end

function plot_various_corr(
    ξ::AbstractVector, 
    ξvind::AbstractVector, 
    Pξ_abs::AbstractVector)

    normed_Pξ_abs = Pξ_abs ./ maximum(Pξ_abs)

    Pfz_anal = plotpeaks(ξ*um, normed_Pξ_abs; 
                            peaks=ξvind, 
                            prominences=true, widths=true, 
                            # xlim=(-0.2, 0.2),
                            # ylim=(0,ymax),
                            # ylim=(10^-34, -10^-1),
                            # annotations=anno)
                        )

    Pfz_anal_log = plotpeaks(ξ*um, normed_Pξ_abs; 
                            peaks=ξvind,
                            # prominences=true, widths=true, 
                            # xlim=(-0.2, 0.2),
                            # ylim=(0,ymax),
                            yscale=:log10,  
                            # ylim=(10^-34, -10^-1),
                            # annotations=anno)
                        )
    Pfz_anal_log_ind = plotpeaks(ξ*um, normed_Pξ_abs; 
                            peaks=ξvind,
                            prominences=true, widths=true, 
                            # xlim=(-0.2, 0.2),
                            # ylim=(0,ymax),
                            yscale=:log10,  
                            # ylim=(10^-34, -10^-1),
                            # annotations=anno)
                        )

    plots = [Pfz_anal, Pfz_anal_log, Pfz_anal_log_ind]

    layout = @layout [grid(3,1)]

    allplots = plot(plots..., 
                    layout=layout, 
                    size=(600, 800), 
                    link=:x,
                    left_margin=20px, 
                    bottom_margin=20px,
                    dpi=300)

    return allplots
end

function get_h(
    Lx::Number,
    Lz::Number,
    α::Number,
    mode_num::Int,
    Efield::Matrix{ComplexF64},
    nref::Number,
    n::Matrix{ComplexF64},
    λ::Number,
    ξv::AbstractVector,
    pol::String,
    pml::Bool;
    savedir::String="./",
    ymax::Number=Inf
    )::AbstractVector

    Nx = size(Efield, 1)
    Nz = size(Efield, 2)

    x = range(-Lx/2, Lx/2; length=Nx)
    z = range(0, Lz; length=Nz)

    dx = step(x)
    dz = step(z)

    # Since we inverted the sign of ξv, 
    # as FFTW returns -ξ, not ξ, 
    # roll back the sign of ξv.
    ξv = -ξv

    k0 = 2*π/λ
    β = nref*k0
    βs = ξv .+ β
    neff = (β.-ξv)./k0

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

                phasor = exp.(-1im*ξv[μ]*z) # since ξv is negetive here, minus sign is added.
                phasor_mat = repeat(transpose(phasor), Nx, 1)
                integrand = psi .* dz .* phasor_mat

                Δβ = ξv[μ]-ξv[v]
                lim = 2*π / abs(Δβ)
                ind = argmin(abs.(z .- lim))
                h = vec(sum(integrand[:,1:ind], dims=2)) # get rid of mode v from integrand.
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

                psi = get_Efield(x, z, 
                                    nref, n, λ, α, h,
                                    pol, pml)
                push!(psifields, psi)

                figname = "$nametag-after-propagation.png"
                corr_h, ξ_h, ξvind_h, ξv_h, peakh_h, Pξ_abs_h = correlation_method(psi, dx, dz)
                profileplots = plot_analysis(x, z, psi, n, h, corr_h, 
                                                ξ_h, ξv_h, ξvind_h, 
                                                peakh_h, Pξ_abs_h; 
                                                )

                layout = @layout [grid(1,3); b{0.333w} c{0.666w}]
                allplots = plot(profileplots..., layout=layout, size=(1400, 1000))
                savefig(allplots, savedir*figname)

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
    nref::Number
    )::AbstractVecOrMat{T} where T

    num = length(mode_profiles)

    # labels = ["input"; ["mode $(i-1)" for i in 1:num]]

    k0 = 2*π / λ
    β = k0 * nref
    labels = ["mode $(i-1), Neff$i=$(@sprintf("%.7f", (ξv[i]+β)/k0))" for i in 1:num]

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

    return plots
end

function get_mode_profiles_im_dis(
    x::AbstractVector,
    τ::AbstractVector,
    uline::AbstractVector,
    pol::String,
    n::AbstractMatrix, 
    ntrial::Number,
    λ::Number,
    α::Number;
    iternum::Number=2,
    modenum = 0,
    figsave = true,
    nametag::String="",
    savedir = "./"
    )

    dx = step(x)
    dτ = step(τ)
    k = 2*π / λ
    # weightedβ = ntrial*k
    newinput = uline

    # Get mode 0.
    for i in 1:iternum
        ufield = get_Efield(x, τ, 
                            ntrial, n, λ, α, 
                            newinput, 
                            pol,
                            false
                            )

        weightedξv = real.((log.(vec(sum(ufield[:,2:end  ], dims=1)).*dx) - 
                            log.(vec(sum(ufield[:,1:end-1], dims=1)).*dx)).*(1im/dτ))

        # @show size(ufield[:,2:end])
        # @show size(weightedξv)
        # if weightedξv > 0
            # println("Numerical error stacked up! \
                    # Choose ntrial \
                    # lower than the one you are using.")
            # break
        # end
        weightedβ = weightedξv .+ (ntrial*k)

        zlast = real(τ[end]*(-1im))
        af = ufield[:,end] ./ exp(weightedξv[end]*zlast)

        modeweight = sqrt(sum(af .* conj.(af)))
        modefunc = af ./ modeweight

        figname = "$nametag-get-mode_$modenum-trial_$i.png"

        plot_im_dis(x, τ, 
                    newinput, 
                    ufield, 
                    modenum,
                    modeweight, 
                    modefunc,
                    ntrial,
                    k,
                    weightedβ,
                    figname; 
                    savedir=savedir,
                    figsave=figsave,
                    )

        # inputplot, Iplot, nplot = plot_field(x, τ, ufield, n, newinput)
        # layout = @layout [grid(1,3)]
        # allplots = plot([inputplot, nplot, Iplot]..., layout=layout, size=(1400, 500))
        # savefig(allplots, savedir*figname)

        st = @sprintf( "Trial %2d: \
                        ntrial=%9.6f, \
                        weightedξv/k=%9.6f, \
                        redefine ntrial as:%9.6f\n",
                        i,
                        ntrial, 
                        weightedξv[end]/k, 
                        weightedβ[end]/k
                        )
 
        print(st)

        ntrial = weightedβ[end]/k
        newinput .-= af
    end

    return newinput
end # function end.

function plot_im_dis(
    x, 
    τ, 
    uline::AbstractVector, 
    ufield::AbstractMatrix,
    modenum::Int,
    modeweight::Number,
    modefunc::AbstractVector,
    ntrial::Number,
    wavenum::Number,
    weightedβ::AbstractVector,
    figname;
    figsave=true,
    savedir="./",
    )

    # @show size(weightedβ)
    intensity = abs2.(ufield)
    input_abs = (abs.(uline).^2) 
    effindex = weightedβ[end]/wavenum

    # normed_input_abs = input_abs./maximum(input_abs)
    weighted_mode = modeweight.*modefunc
    tit = "\$a_$(modenum)f_$(modenum)(x,y)\$"
    nst = @sprintf("ntrial=%8.6f", ntrial)

    inplot = plot(x./um, input_abs, 
                            label="input beam", 
                            title="Input, $nst",
                            xlabel="x (μm)",
                            lw=1.5, dpi=300,)

    irplot = heatmap(abs.(τ)./um, x./um, intensity, 
                    dpi=300, 
                    clim=(0,Inf), 
                    c=:thermal, 
                    xlabel="z (μm)", 
                    ylabel="x (μm)", 
                    label="Intensity", 
                    title="Straight waveguide")

    afplot = plot(x./um, 
                    real.(weighted_mode),
                    label="\$β_$(modenum)/k=$(effindex)\$", 
                    title=tit,
                    xlabel="x (μm)",
                    lw=1.5, 
                    dpi=300
                    )

    hplot = plot(   x./um, 
                    real.(uline .- weighted_mode), 
                    title="Input - \$a_$(modenum)f_$(modenum)(x,y), \$",
                    xlabel="x (μm)",
                    lw=1.5, 
                    dpi=300
                    )

    βplot = plot(   abs.(τ[2:end]) ./ um,
                    weightedβ ./ wavenum,
                    label="\$β/k→$(weightedβ[end]/wavenum)\$",
                    title="propagation constant convergence",
                    xlabel="τ",
                    ylabel="Effective index"
                )
    # fplot = plot(x./um, 
    #                 real.(modefunction), 
    #                 label="f(x,∞)", 
    #                 title="\$f_$(iternum-1)(x,y), \$"*nst,
    #                 xlabel="τ",
    #                 ylabel="x (μm)",
    #                 lw=1.5, 
    #                 dpi=300
    #                 )
    # annotate!(fplot, 5, -0.2, text("\$a_$(iternum-1)=$weight", 12, :blue))
    plots = [afplot, βplot, inplot, irplot, hplot]
    layout = @layout [grid(2,3)]
    plot(plots..., 
            layout=layout, 
            size=(1500, 1000)
        )

    if figsave == true
        savefig(savedir*figname)
    end

end # function end.

end # module end.