using VideoIO
using Images
using ProgressMeter
include("./ColorCurves.jl")
include("./Mandelbrot.jl")

T = 2
framerate = 30
N = T * framerate
period = 0.3e5

starting_radius = 0.003

x = range(-1.4002, -1.4011, N)
r = range(starting_radius, starting_radius/21.78, N)

color_curve = default_cosine_curve

open_video_out("looping.gif", RGB{N0f8}, (2000, 2000), framerate = framerate) do writer
    @showprogress "writing video frames..." for i in 1:N

        center = BigFloat(x[i]) + BigFloat(0.0)im
        radius = r[i]
        
        μ = perturbation_pipeline(
            center, radius, (2000, 2000), 2, 500_000, reference_point = BigFloat(0.0) + BigFloat(0.0)im,
            renderer = naive_interior_renderer, gpu = true)
        
        write(writer, RGB{N0f8}.(color_curve.(μ, period)))
    end     
end

CUDA.reclaim()