module FD_SVBPM_2D

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

export  get_gaussian_input, 
        get_rib_waveguide_profile,
        get_step_index_profile, 
        get_S_bend_profile,
        get_symmetric_Gaussian_index_profile,
        _to_next_z, 
        _pml, 
        get_Efield, 
        correlation_method, 
        plot_field, 
        plot_with_corr, 
        get_h, 
        plot_mode, 
        get_mode_profiles_im_dis

function get_gaussian_input(
    x::AbstractVector, 
    y::AbstractVector, 
    xshift::Number, 
    yshift::Number, 
    wx::Number,
    wy::Number;
    plot=false,
    savedir="./"
    )

    X = reshape(x, :, 1) .* ones(1, length(y))
    Y = reshape(y, 1, :) .* ones(length(x), 1)

    Eplane = exp.(-(((X.-xshift).^2) ./ (wx^2)) .- ((Y.-yshift).^2 ./ (wy^2))) .+ 0im
    # @show X[:,1]
    # @show Y[1,:]
    # @show size(Eplane)

    if plot == true

        to_plot = real.(Eplane)

        nmax = maximum(to_plot)
        nmin = minimum(to_plot)

        hm = heatmap(y./um, x./um, to_plot;
                        dpi=300, 
                        clim=(nmin, nmax), 
                        # colormap=:tab20,
                        xlabel="y (μm)", 
                        ylabel="x (μm)", 
                        title="Input beam profile")

        savefig(savedir*"input_beam_profile.png")
    end

    return Eplane, hm
end

function get_S_bend_profile(
    x::AbstractVector,
    y::AbstractVector,
    z::AbstractVector,
    nc::Number,
    ns::Number,
    Δn::Number,
    d::Number,
    w::Number;
)
    
    Nx = length(x)
    Ny = length(y)
    Nz = length(z)

    dx = step(x)
    dy = step(y)
    dz = step(z)

    n = ones(ComplexF64, Nx, Ny, Nz) .* nc

    @assert Nx % 2 == 1 # To include x=0.
    @assert Ny % 2 == 1 # To include y=0.

    for i in 1:Nx
        for j in 1:Ny

            if x[i] < 0

                if y[j] < -w/2
                    n[i,j,:] .= ns + (Δn * exp(-(x[i]/d)^2) * exp(-((y[j]+(w/2))/d)^2))
                elseif y[j] < w/2 && y[j] > -w/2
                    n[i,j,:] .= ns + (Δn * exp(-(x[i]/d)^2))
                elseif y[j] > w/2
                    n[i,j,:] .= ns + (Δn * exp(-(x[i]/d)^2) * exp(-((y[j]-(w/2))/d)^2))
                end

            end
        end
    end

    return n
    
end

function get_rib_waveguide_profile(
    x::AbstractVector,
    y::AbstractVector,
    z::AbstractVector,
    w1::Number,
    w2::Number,
    h::Number,
    d::Number,
    n1::Number,
    n2::Number,
    n3::Number;
    )

    Nx = length(x)
    Ny = length(y)
    Nz = length(z)

    dx = step(x)
    dy = step(y)
    dz = step(z)

    n = ones(ComplexF64, Nx, Ny, Nz) .* n1

    @assert Nx % 2 == 1 # To include x=0.
    @assert Ny % 2 == 1 # To include y=0.

    cx = Int(ceil(Nx/2))
    sx = cx + Int(round(d/dx))
    ux = sx + Int(round(h/dx))

    cy = Int(ceil(Ny/2))
    sy = (cy-Int(round(w1/2/dy))):(cy+Int(round(w1/2/dy)))
    uy = (cy-Int(round(w2/2/dy))):(cy+Int(round(w2/2/dy)))

    n[ 1:cx, uy, :] .= n3
    n[cx:sx, uy, :] .= n2
    n[sx:ux, sy, :] .= n2

    return n
    
end

function get_step_index_profile(
    x::AbstractVector, 
    z::AbstractVector, 
    loc_discont::AbstractVector,
    refractiveindice::AbstractVector;
    save = false,
    savedir = "./"
    )::Array{Number, 3}

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
    y::AbstractVector,
    z::AbstractVector,
    nref::Number, 
    n::Array{<:Number, 3}, 
    λ::Number, 
    α::Number, 
    input::Matrix{<:Number};
    pmlx = false,
    pmly = false,
    )::Array{<:Number, 3}

    npml = 10

    Nx = length(x)
    Ny = length(y)
    Nz = length(z)

    dx = step(x)
    dy = step(y)
    dz = step(z)

    Efield = zeros(ComplexF64, Nx, Ny, Nz)
    Efield[:,:,1] = input

    if pmlx == true && pmly == false
        Py, Qy, Ry, Fx, Gx, Hx = _pml(Nx, Ny, Nz, npml, dx, dy, n, λ; pmlx=true, pmly=false)
    elseif pmlx == false && pmly == true
        Py, Qy, Ry, Fx, Gx, Hx = _pml(Nx, Ny, Nz, npml, dx, dy, n, λ; pmlx=false, pmly=true)
    elseif pmlx == true && pmly == true
        Py, Qy, Ry, Fx, Gx, Hx = _pml(Nx, Ny, Nz, npml, dx, dy, n, λ; pmlx=true, pmly=true)
    else
        Py, Qy, Ry, Fx, Gx, Hx = _pml(Nx, Ny, Nz, npml, dx, dy, n, λ; pmlx=false, pmly=false)
    end

    for step in 2:Nz
        adi = Efield[:,:,step-1]
        adi = _1st_sub_step_scalar(step, λ, dy, dz, nref, n, Py, Qy, Ry, adi;)
        adi = _2nd_sub_step_scalar(step, λ, dx, dz, nref, n, Fx, Gx, Hx, adi;)
        Efield[:,:,step] = adi
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
    Ny::Int64,
    Nz::Int64,
    npml::Int64,
    dx::Number,
    dy::Number,
    n::Array{<:Number, 3}, 
    λ::Number;
    pmlx = false,
    pmly = false 
    )

    rc0 = 1.e-16
    μ0 = 4*π*10^-7
    ε0 = 8.8541878128e-12
    c = sqrt(1/μ0/ε0)
    imp = sqrt(μ0/ε0)
    ω = 2*π*c / λ

    go = 2. # grading order.
    bdwx = (npml-1) * dx
    bdwy = (npml-1) * dy
    σx_max = -(go+1) * log(rc0) / (2*imp*bdwx)
    σy_max = -(go+1) * log(rc0) / (2*imp*bdwy)
    # σmax = 5*ε0*ω

    loc = range(0, 1, length=npml)
    σx = σx_max .* (loc.^go)
    σy = σy_max .* (loc.^go)

    σx = reshape(σx , npml, 1, 1)
    σy = reshape(σy , 1, npml, 1)

    nxll = n[      npml:-1:  1,:,:]
    nxrr = n[end+1-npml: 1:end,:,:]

    nyll = n[:,      npml:-1:  1,:]
    nyrr = n[:,end+1-npml: 1:end,:]

    qx = ones(ComplexF64, Nx, Ny, Nz)
    qy = ones(ComplexF64, Nx, Ny, Nz)

    if pmlx == true 
        qx[      npml:-1:  1, :, :] = 1 ./ (1 .- ((1im.*σx) ./ (ω .* ε0 .* nxll.^2)))
        qx[end+1-npml: 1:end, :, :] = 1 ./ (1 .- ((1im.*σx) ./ (ω .* ε0 .* nxrr.^2)))
    end

    if pmly == true 
        qy[:,       npml:-1:  1, :] = 1 ./ (1 .- ((1im.*σy) ./ (ω .* ε0 .* nyll.^2)))
        qy[:, end+1-npml: 1:end, :] = 1 ./ (1 .- ((1im.*σy) ./ (ω .* ε0 .* nyrr.^2)))
    end

    qxp = qx[2:end  ,:,:]
    qxm = qx[1:end-1,:,:]

    # nxp = n[2:end  ,:,:]
    # nxm = n[1:end-1,:,:]

    qyp = qy[:,2:end  ,:]
    qym = qy[:,1:end-1,:]

    nyp = n[:,2:end  ,:]
    nym = n[:,1:end-1,:]

    Ry = zeros(ComplexF64, Nx, Ny, Nz) 
    Hx = zeros(ComplexF64, Nx, Ny, Nz) 

    qyn = (qyp .+ qym)./(nyp.^2 .+ nym.^2)

    Py = qyp .* qyn .* (nym.^2)
    Qy = qym .* qyn .* (nyp.^2)

    Ry[:,2:end  ,:] .+= qyp .* qyn .* (nyp.^2)
    Ry[:,1:end-1,:] .+= qym .* qyn .* (nym.^2)

    Fx = qxp .* (qxp .+ qxm) ./ 2
    Gx = qxm .* (qxm .+ qxp) ./ 2

    Hx .+= qx.^2
    Hx[2:end  ,:,:] += qxp .* qxm ./ 2
    Hx[1:end-1,:,:] += qxm .* qxp ./ 2

    return Py, Qy, Ry, Fx, Gx, Hx

end

"""
function _1st_sub_step_scalar
    In this function, we used 'exp(i(wt-kr))' notation.

# Arguments
# Returns
# Example
"""
function _1st_sub_step_scalar(
    step::Int,
    λ::Number,
    dy::Number,
    dz::Number,
    nref::Number,
    n::Array{<:Number, 3},
    P::Array{<:Number, 3},
    Q::Array{<:Number, 3},
    R::Array{<:Number, 3},
    E::Matrix{ComplexF64};
    )::Matrix{ComplexF64}

    Nx = size(n,1)
    Ny = size(n,2)

    u_half_step = zeros(ComplexF64, Nx, Ny)

    for i in 1:Nx
        k0 = 2*π / λ
        nd = n[i,:,step].^2 .- (nref^2)

        aj = (-P[i,:,step] ./ (dy^2))
        bj = (4im*k0*nref/dz) .+ (R[i,:,step] ./ (dy^2)) .- (0.5 .* k0^2 .* nd)
        cj = (-Q[i,:,step] ./ (dy^2))
        Aj = diagm(-1=>aj, 0=>bj, 1=>cj)

        Dj = (4im*k0*nref/dz) .- (R[i,:,step] ./ (dy^2)) .+ (0.5 .* k0^2 .* nd)
        above = (1 / dy^2) * Q[i,:,step]
        below = (1 / dy^2) * P[i,:,step]

        Bj = diagm(-1=>below, 0=>Dj, 1=>above)

        # @show size(aj)
        # @show size(bj)
        # @show size(cj)
        # @show size(Aj)
        # @show size(Dj)
        # @show size(Bj)
        # @show size(E[i,:])

        r = Bj * E[i,:]
        u_half_step[i,:] = Aj \ r
    end

    return u_half_step
end

function _2nd_sub_step_scalar(
    step::Int,
    λ::Number,
    dx::Number, 
    dz::Number, 
    nref::Number,
    n::Array{<:Number, 3}, 
    F::Array{<:Number, 3}, 
    G::Array{<:Number, 3}, 
    H::Array{<:Number, 3}, 
    E::Matrix{ComplexF64};
    )::Matrix{ComplexF64}

    Nx = size(n,1)
    Ny = size(n,2)

    u_next_step = zeros(ComplexF64, Nx, Ny)

    for j in 1:Ny
        k0 = 2*π / λ
        nd = n[:,j,step].^2 .- nref^2

        ai = (-1/dx^2) .* F[:,j,step]
        bi = (4im*k0*nref/dz) .+ (H[:,j,step] ./ dx^2) .- (0.5 .* k0^2 .* nd)
        ci = (-1/dx^2) .* G[:,j,step]
        Ai = diagm(-1=>ai, 0=>bi, 1=>ci)

        Di = (4im*k0*nref/dz) .- (H[:,j,step] ./ dx^2) .+ (0.5 .* k0^2 .* nd)
        above = (1 / dx^2) .* G[:,j,step]
        below = (1 / dx^2) .* F[:,j,step]

        Bi = diagm(-1=>below, 0=>Di, 1=>above)

        r = Bi * E[:,j]
        u_next_step[:,j] = Ai \ r
    end

    return u_next_step
end


function plot_field(
    x::AbstractVector, 
    y::AbstractVector, 
    z::AbstractVector, 
    field::AbstractArray{<:Number, 3},
    which_plane::String,
    intercept::Number;
    clim::Tuple{Number, Number} = (-Inf, Inf),
    )

    intensity = abs2.(field)

    if which_plane == "xy" || which_plane == "yx"
        ind = argmin(abs.(z .- intercept))
        slice = intensity[:,:,ind]
        interceptaxis = 'z'
        xaxis = y./um
        yaxis = x./um
        xlabel= "y (μm)"
        ylabel= "x (μm)"

    elseif which_plane == "xz" || which_plane == "zx"
        ind = argmin(abs.(y .- intercept))
        slice = intensity[:,ind,:]
        interceptaxis = 'y'
        xaxis = abs.(z)./um
        yaxis = x./um
        xlabel= "z (μm)"
        ylabel= "x (μm)"

    elseif which_plane == "yz" || which_plane == "zy"
        ind = argmin(abs.(x .- intercept))
        slice = intensity[ind,:,:]
        interceptaxis = 'x'
        xaxis = abs.(z)./um
        yaxis = y./um
        xlabel= "z (μm)"
        ylabel= "y (μm)"
        # @show size(intensity)
        # @show size(slice)
        # @show size(xaxis)
        # @show size(yaxis)
    end

    title = "$which_plane plane at $interceptaxis=$(intercept/um) μm"
    power = heatmap(xaxis, yaxis, slice,
                    dpi=300, clim=clim, c=:thermal,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    title=title)

    return power
end


"""
function correlation_method

    Note that Pf is not a function of ξ.
    Note that since fftfreq returns frequency, 
    we need 2*π to make angular freq.
"""
function correlation_method(
    Efield::Matrix, 
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

function plot_with_corr(
    x::AbstractVector,
    z::AbstractVector, 
    field::Matrix, 
    n::Matrix{ComplexF64},
    input::AbstractVector,
    Pz::AbstractVector,
    ξ::AbstractVector,
    ξv::AbstractVector,
    ξvind::AbstractVector,
    peakh::AbstractVector,
    Pξ_abs::AbstractVector;)

    Iplot= plot_field(x, y, z, field, "xz", 0)

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

function get_h(
    Lx::Number,
    Lz::Number,
    α::Number,
    mode_num::Int,
    Efield::Matrix{ComplexF64},
    nref::Number,
    n::Matrix{ComplexF64},
    λ::Number,
    ξv::AbstractVector;
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

                psi = get_Efield(x, y, z, nref, n, λ, α, h)
                push!(psifields, psi)

                figname = "$nametag-after-propagation.png"
                corr_h, ξ_h, ξvind_h, ξv_h, peakh_h, Pξ_abs_h = correlation_method(psi, dx, dz)
                profileplots = plot_with_corr(x, z, psi, n, h, corr_h, 
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
    mode_profiles::VecOrMat{T},
    ξv::AbstractVector,
    λ::Number,
    nref::Number
    )::VecOrMat{T} where T

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
    n::Matrix, 
    ntrial::Number,
    λ::Number,
    α::Number,
    iternum::Number;
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
        ufield = get_Efield(x, y, τ, ntrial, n, λ, α, newinput; pml=false)
        af = ufield[:,end]

        weightedξv = real((log(sum(af)*dx) - log(sum(ufield[:,end-1])*dx))*1im/dτ)
        weightedβ += weightedξv

        figname = "get_mode_$mode-trial_$i.png"
        plot_im_dis(x, τ, newinput, ufield, af, ntrial, figname; 
                    savedir=savedir,
                    figsave=figsave)

        st = @sprintf("ntrial=%9.6f, weightedξv/k=%9.6f, redefined n =%9.6f\n", 
                        ntrial, weightedξv/k, weightedβ/k)
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
    af,
    ntrial,
    figname;
    figsave=true,
    savedir="./",
    )

    intensity = abs2.(ufield)
    input_abs = (abs.(uline).^2) 

    normed_input_abs = input_abs./maximum(input_abs)
    iplot = plot(x./um, input_abs, 
                            label="input beam", 
                            title="Input",
                            xlabel="z (μm)",
                            ylabel="x (μm)",
                            lw=1.5, dpi=300,)

    Eplot = heatmap(abs.(τ)./um, x./um, intensity, 
                    dpi=300, 
                    clim=(0,Inf), 
                    c=:thermal, 
                    xlabel="z (μm)", ylabel="x (μm)", zlabel="Intensity", 
                    title="Straight waveguide")

    nst = @sprintf("ntrial=%8.6f", ntrial)
    fplot = plot(x./um, 
                    real.(af),
                    label="f(x,∞)", 
                    title="Output,"*nst,
                    xlabel="τ",
                    ylabel="x (μm)",
                    lw=1.5, 
                    dpi=300)
    # annotate!(fplot, 5, -0.2, text(nst, 12, :blue))
    plots = [iplot, Eplot, fplot]
    layout = @layout [grid(1,3)]
    plot(plots..., layout=layout, size=(1400, 500))

    if figsave == true
        savefig(savedir*figname)
    end

end # function end.

end # module end.