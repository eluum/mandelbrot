setprecision(BigFloat, 8*2048)
include("Mandelbrot.jl")

l = locations["julia"]
nearby = l.center 
         
max_iter = 10_000_000
resolution = (4000, 4000)
#radius = ExtendedRangeFloat{Float32, Int32}(0.7, -8664)
radius = l.radius

# pixel tolerance
ϵ = BigFloat(radius) / minimum(resolution)

(period, _) = atom_domain(nearby, max_iter)
println(period)

refined = atom_nucleus(nearby, period; ϵ)
println(refined)