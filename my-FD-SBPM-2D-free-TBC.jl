using Plots
using LinearAlgebra

um = 10^-6
nm = 10^-9

Lx = 5*um
Lz = 20*um

Nx = 100
Nz = 300

dx = Lx / (Nx-1)
dz = Lz / (Nz-1)

h = dz

λ = 500*nm
w = 2*um

n = 1
k0 = 2*π / λ
beta = n*k0

x = range(-Lx/2, Lx/2; length=Nx)
z = range(0, Lz; length=Nz)

Eline1 = exp.(-(x./w).^2)
Eline2 = exp.(-(x./w).^2)

Efield1 = zeros(ComplexF64, Nx, Nz)
Efield2 = zeros(ComplexF64, Nx, Nz)
Efield1[:,1] = Eline1
Efield2[:,1] = Eline2

"""
function get_E(dx, h, beta, E; tbc::Bool=true)

In this function, we have used `exp(i(wt-kr))` notation.

# Arguments
- `dx::Float64`: spacing along x-axis.
- `h::Float64` : increment along z-axis.
- `beta::Float64` : propagation parameter along z-axis.
- `E::Vector{ComplexF64}` : scalar electric field on xy-plane.
- `tbc`::Bool : TBC on if true.

# Returns
- `new_zplane::Vector{ComplexF64}`: scalar electric field on xy-plane.
"""
function get_E(
    dx::Float64, 
    h::Float64, 
    beta::Float64, 
    E::Vector{ComplexF64}; 
    tbc::Bool=true)::Vector{ComplexF64}

    main = ones(Nx) .* -2
    above = ones(Nx - 1)
    below = ones(Nx - 1)
    P = diagm(-1=>below, 0=>main, 1=>above)
    P = P .* 1/(2*beta*dx^2) 

    Lp = I + 0.5im*h.*P
    Lm = I - 0.5im*h.*P

    if tbc == true
        # Implementation of boundary conditions.
        kx = 1im/dx * log(E[2]/E[1])

        # On the left.
        left = 0.5im * h / (2*beta) / dx^2 * exp(1im * kx * dx)
        Lp[1,1] += left
        Lm[1,1] -= left

        # On the right.
        kx = -1im/dx * log(E[end]/E[end-1])
        right = 0.5im * h / (2*beta) / dx^2 * exp(1im * kx * dx)
        Lp[end,end] += right
        Lm[end,end] -= right
    end

    new_zplane = Lp \ (Lm * E)

    return new_zplane
end

for k in axes(Efield1, 2)[2:end]
    global Eline1
    Eline1 = get_E(dx, h ,beta, Eline1; tbc=false)
    Efield1[:,k] = Eline1
end

for k in axes(Efield2, 2)[2:end]
    global Eline2
    Eline2 = get_E(dx, h ,beta, Eline2; tbc=true)
    Efield2[:,k] = Eline2
end

intensity1 = abs2.(Efield1)
intensity2 = abs2.(Efield2)

# Plot irradiance.
hm1 = heatmap(intensity1, dpi=300, clim=(0,1), c=:thermal, xlabel='x', ylabel='z', zlabel="Intensity", title="TBC off")
hm2 = heatmap(intensity2, dpi=300, clim=(0,1), c=:thermal, xlabel='x', ylabel='z', zlabel="Intensity", title="TBC on")

plots = [hm1, hm2]
layout = @layout [a b]
plot(plots..., layout=layout, size=(1200, 600))

savefig("./my-FD_SBPM-2D-free-TBC.png")