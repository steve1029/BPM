using Plots
using LinearAlgebra

um = 10^-6
nm = 10^-9

Lx = 10*um
Lz = 20*um

Nx = 60
Nz = 300

dx = Lx / (Nx-1)
dz = Lz / (Nz-1)

h = dx

λ = 600*nm
w = 3*um

n = 1
k0 = 2*π / λ
beta = n*k0

x = range(-Lx/2, Lx/2; length=Nx)
z = range(0, Lz; length=Nz)

Efield = zeros(ComplexF64, Nx, Nz)
Eline = exp.(-(x./w).^2)
Efield[:,1] = Eline

# Plot E0
input_beam_plot = plot(x, abs.(Eline).^2, label="input beam", lw=1.5, dpi=300, size=(500,500))
xlabel!("x")
ylabel!("Intensity of input beam")
savefig("./beam_input.png")

function get_E(dx, h, beta, E)

    main = ones(Nx) .* -2
    above = ones(Nx - 1)
    below = ones(Nx - 1)
    P = diagm(-1=>below, 0=>main, 1=>above)
    P = P .* 1/(2*beta*dx^2) 

    Lp = I - 0.5im*h.*P
    Lm = I + 0.5im*h.*P

    new_zplane = Lp \ (Lm * E)

    return new_zplane
end

for k in axes(Efield, 2)[2:end]
    global Eline
    Eline = get_E(dx, h ,beta, Eline)
    Efield[:,k] = Eline
end

intensity = abs2.(Efield)

# Plot irradiance.
heatmap(intensity, size=(800,800), dpi=300, clim=(0,1), c=:thermal)
xlabel!("x")
ylabel!("z")
zlabel!("Intensity")
savefig("./my-FD_SBPM-2D-noTBC.png")