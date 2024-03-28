using VideoIO
using ProgressMeter
include("./ColorCurves.jl")

T = 60
framerate = 30

period_range = range(1.0e-3, 1.0e6, T * framerate)

#color_curve = CircleColorCurve(0.6, 1.0)
color_curve = default_cosine_curve

open_video_out("testgif.gif", RGB{N0f8}, (2000,2000), framerate = 30) do writer
    @showprogress "writing video frames..." for period in period_range 
        write(writer, RGB{N0f8}.(color_curve.(μ, period)))
    end     
end