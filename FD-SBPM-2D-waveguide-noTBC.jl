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
α = 0.5001

x = range(-Lx/2, Lx/2; length=Nx)
z = range(0, Lz; length=Nz)

Eline = exp.(-(x./w).^2) .+ 0im
Efield = ones(ComplexF64, Nx, Nz)
Efield[:,1] = Eline

n = ones(ComplexF64, Nx, Nz)
up = Int(round(Nx/2 - Nx/6))
down = Int(round(Nx/2 + Nx/6))
n[up:down,:] .= 2

n0 = 1.0

function to_next(
    step::Int64,
    α::Float64,
    λ::Float64,
    dx::Float64, 
    dz::Float64, 
    n0::Float64, 
    n::Matrix{ComplexF64}, 
    E::Vector{ComplexF64}
    )::Vector{ComplexF64}

    k0 = 2*π / λ

    nd = n[:,step].^2 .- n0^2

    a = ones(Float64, Nx-1) .* (-α/dx^2)
    b = ones(ComplexF64, Nx) .* ((2*α/dx^2) .- (α.*nd.*k0^2) .+ (2im*k0*n0/dz))
    c = a
    tri = diagm(-1=>c, 0=>b, 1=>a)

    A = ones(ComplexF64, Nx) .* ((1-α).*nd.*k0^2 .- (2*(1-α)/dx^2) .+ (2im*k0*n0/dz))
    above = ones(Float64, Nx-1) .* ((1-α) / dx^2)
    below = above

    B = diagm(-1=>below, 0=>A, 1=>above)

    r = B * E

    newE = tri \ r

    return newE

end

for k in axes(Efield, 2)[2:end]
    global Eline
    Eline = to_next(k, α, λ, dx, dz, n0, n, Eline)
    Efield[:,k] = Eline
end

intensity = abs2.(Efield)

hm1 = heatmap(intensity, dpi=300, clim=(0,1), c=:thermal, xlabel='x', ylabel='z', zlabel="Intensity", title="Straight waveguide")
hm2 = heatmap(real(n), dpi=300, clim=(1,2), xlabel='x', ylabel='z', zlabel="index", title="Refractive index")

plots = [hm1, hm2]
layout = @layout [a b]
plot(plots..., layout=layout, size=(1200, 600))

savefig("./FD_SBPM-2D-waveguide-TBC.png")