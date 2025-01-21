# File name: bpm_tbc.jl
# Illustrates propagation of Gaussian pulse in a free space
# using BPM with transparent boundary conditions 

using FFTW
using LinearAlgebra
using Plots
using Printf
using Ranges

# Parameters
L_x = 10.0        # transversal dimension (along x-axis)
w_0 = 1.0         # width of input Gaussian pulse
lambda = 0.6      # wavelength
n = 1.0           # refractive index of the medium
k_0 = 2 * π / lambda # wavenumber

# Discretization
N_x = 128              # number of points on x-axis
Delta_x = L_x / (N_x - 1)  # x-axis spacing
h = 5 * Delta_x        # propagation step
N_z = 100              # number of propagation steps

# Coordinates and initial field
x = range(-0.5*L_x, stop=0.5*L_x, length=N_x)
E = exp.(-(x./ w_0).^2)   # initial Gaussian field

# Initialize storage for plotting
plotting = zeros(N_x, N_z)
z_plot = zeros(N_z) # propagation distance for plotting

@show typeof(plotting)
@show typeof(z_plot)

# BPM stepping function
function step(Delta_x, k_0, h, n, E_old)
    # Function propagates BPM solution along one step
    N_x = size(E_old, 1) # determine size of the system
    
    # Defines operator P outside of a boundary
    prefactor = 1 / (2 * n * k_0 * Delta_x^2)
    main = ones(N_x)
    above = ones(N_x - 1)
    below = ones(N_x - 1)
    
    P = prefactor * (diagm(-1 => above) - 2 * diagm(0 => main) + diagm(1 => below)) # matrix P

    L_plus = I + 0.5im * h * P # step forward
    L_minus = I - 0.5im * h * P # step backward

    # Implementation of boundary conditions
    pref = 0.5im * h / (2 * k_0 * Delta_x^2)

    k = 1im / Delta_x * log(E_old[2] / E_old[1])
    if real(k) < 0
        k = 1im * imag(k)
    end

    left = pref * exp(1im * k * Delta_x) # left correction for next step
    L_plus[1,1] += left
    L_minus[1,1] -= left

    k = -1im / Delta_x * log(E_old[end] / E_old[end - 1])
    if real(k) < 0
        k = 1im * imag(k)
    end
    right = pref * exp(1im * k * Delta_x) # right correction for next step
    L_plus[end,end] += right
    L_minus[end,end] -= right

    E_new = L_minus \ (L_plus * E_old) # determine new solution
    
    return E_new
end

# BPM stepping
for r in 1:N_z
    z_plot[r] = r * h
    plotting[:, r] = abs.(E).^2
    global E
    E = step(Delta_x, k_0, h, n, E) # Propagates pulse over one step
end

# 2D plots every 10-th step
plot() # This function is in Plots.jl. It Initialize the new plot.
for k in 1:10:N_z
    plot!(x, plotting[:, k], label="$k", lw=1.5, dpi=300, size=(800, 400))
    # savefig("./sample_2d_$k.png")
end
xlabel!("x")
ylabel!("Intensity")
savefig("./Wartak_2d.png")

# 3D plots every 10-th step
plot3d(x, zeros(N_z), plotting[:, 1], size=(800, 800), dpi=300)
for k in 1:10:N_z
    y = z_plot[k] * ones(N_x)
    plot3d!(x, y, plotting[:, k], label="$k", lw=1.5)
    # savefig("./sample_3d_$k.png")
end
xlabel!("x")
ylabel!("z")
zlabel!("Intensity")
savefig("./Wartak_3d.png")