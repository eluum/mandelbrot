include("Mandelbrot.jl")

using GLMakie
using FFTW

#dinkydau's coordinates
Re = parse(BigFloat, "-1.99996619445037030418434688506350579675531241540724851511761922944801584242342684381376129778868913812287046406560949864353810575744772166485672496092803920095332")
Im = parse(BigFloat, "+0.00000000000000000000000000000000030013824367909383240724973039775924987346831190773335270174257280120474975614823581185647299288414075519224186504978181625478529")

flake = Re + Im * 1im

flake = nearest - pixel*153 + (pixel*727)im

max_iterations = 60_001

z, ∂c, ∂z₀ = detailed_orbit(flake, max_iterations)

f = Figure()
ax0 = GLMakie.Axis(f[1, 1])
lines!(ax0, log10.(Float64.(abs.(z))))

ax1 = GLMakie.Axis(f[2, 1])
lines!(ax1, log10.(Float64.(abs.(∂c))))

ax2 = GLMakie.Axis(f[3, 1])
lines!(ax2, log10.(Float64.(abs.(∂z₀))))

linkxaxes!(ax0, ax1, ax2)
f

#ϵ = Float64.(all_atom_domains(adjusted_flake, max_iterations))
#N = length(ϵ)
#lines(1:N, log10.(ϵ))
#N_half = Int64((N + 1) / 2)

# frequency analysis
#ϵf = fft(ϵ .- mean(ϵ))
#f = ifftshift(range(-0.5, 0.5, N))
#p = reverse(1 ./ f[1:N_half])

#lines(f[1:N_half], log10.(abs.(ϵf[1:N_half])))

#lags = ifft(ϵf .* conj.(ϵf))
#lines(real.(lags))
