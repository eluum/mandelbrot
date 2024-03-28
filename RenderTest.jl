
setprecision(BigFloat, 8*2048)
include("Mandelbrot.jl")
include("Locations.jl")
l = locations["morph"]

μ = render_pipeline(
    l.location.center, l.location.radius, (2000, 2000), 2, l.iterations, ExtendedRangeFloat{Float32, Int32}(4.0),
    renderer = BLA_renderer,
    gpu = true, datatype = ExtendedRangeFloat{Float32, Int32})

# μ = render_pipeline(
#     l.location.center, ExtendedRangeFloat{Float32, Int32}(0.6124302f0, -20000), (2000, 2000), 2, l.iterations, ExtendedRangeFloat{Float32, Int32}(4.0),
#     renderer = BLA_renderer,
#     gpu = true, datatype = ExtendedRangeFloat{Float32, Int32})

#μ = render_pipeline(
#    l.center, l.radius, (2000, 2000), 2, l.iterations,
#    renderer = BLA_renderer,
#    gpu = false, datatype = Float64)

CUDA.reclaim()

# per pixel stats
histogram(μ[:], yaxis = :log, xlabel = "normalized escape iteration", ylabel = "count")

# image preview
heatmap(μ, colormap = :jet)
