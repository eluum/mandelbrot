#module Mandelbrot
#export locations, reference_orbit, atom_domain, atom_nucleus, 
    
using Images
using ImageFiltering
using Statistics
using CUDA

include("./ColorCurves.jl")
include("./ExtendedRangeFloats.jl")
using .ExtendedRangeFloats

include("./Renderers.jl")


function δ_mesh(
        square_radius :: Real, 
        resolution :: Tuple{Integer, Integer}, 
        datatype :: DataType = Float64
    ) :: Matrix{Complex{Real}}

    pixel = square_radius / minimum(resolution)

    x = range(-pixel*resolution[1], pixel*resolution[1], resolution[1])
    y = range(pixel*resolution[2], -pixel*resolution[2], resolution[2])

    cx = x' .* ones(resolution[2], resolution[1])
    cy = y  .* ones(resolution[2], resolution[1])

    return Matrix{Complex{datatype}}(cx + (cy)im)
end

function δ_mesh(
        square_radius :: ExtendedRangeFloat, 
        resolution :: Tuple{Integer, Integer}, 
        datatype :: DataType = typeof(square_radius)
    ) :: Matrix{Complex{datatype}}

    map(δ_mesh(square_radius.mantissa, resolution, datatype)) do c
        complex(
            normalize(datatype(real(c), square_radius.exponent)),
            normalize(datatype(imag(c), square_radius.exponent)))
    end
end

function reference_orbit(
        c :: Complex{BigFloat}, 
        max_iteration :: I, 
        R :: DataType = Float64, 
        ϵ = eps(zero(R))
    ) :: Vector{Complex{R}} where {I <: Integer}

    iteration = one(I)

    orbit = Vector{Complex{R}}(undef, max_iteration)
    orbit[1] = Complex{R}(c)
    z = c
    ϵ² = ϵ^2

    while iteration < max_iteration
        z = z*z + c

        # cast to storing precision
        Rz = Complex{R}(z)
        if real(Rz)^2 + imag(Rz)^2 > 4.0
            println("   Escaping reference, $iteration iterations")
            return orbit[1:iteration]
        end

        iteration += one(I)

        # store iteration
        orbit[iteration] = Rz
        if real(Rz)^2 + imag(Rz)^2 < ϵ²
            println("   Periodic reference, period is $iteration")
            return orbit[1:iteration]
        end
    end
    println("   Max iteration reference, capped at $max_iteration")
    return orbit
end

function detailed_orbit(
        c :: Complex{R}, 
        max_iteration :: I
    ) :: Tuple{Vector{Complex{R}}, Vector{Complex{R}}, Vector{Complex{R}}} where 
        {R <: Real, I <: Integer}

    z = zero(Complex{R})
    ∂c = zero(Complex{R})
    ∂z₀ = one(Complex{R})

    # space for orbit history
    zn = Vector{Complex{R}}(undef, max_iteration)
    ∂cn = Vector{Complex{R}}(undef, max_iteration)
    ∂z₀n = Vector{Complex{R}}(undef, max_iteration)

    iteration = zero(I)
    while iteration < max_iteration
        ∂c = 2*z*∂c + 1
        z = z*z + c
        ∂z₀ = 2*z*∂z₀

        iteration += 1 

        # orbit history
        zn[iteration] = z
        ∂cn[iteration] = ∂c
        ∂z₀n[iteration] = ∂z₀
    end
    return zn, ∂cn, ∂z₀n
end

function atom_domain(
        c :: Complex{R}, 
        max_period :: I
    ) :: Tuple{I, R} where {R <: Real, I <: Integer}

    period = zero(I)
    closest = zero(I)
    ϵ² = R(4.0)
    z = zero(Complex{R})
    while period < max_period
        z = z*z + c
        period = period + 1

        ϵᵢ² = real(z)^2 + imag(z)^2

        if ϵᵢ² > 4.0
            break
        end

        if ϵᵢ² < ϵ²
            ϵ² = ϵᵢ²
            closest = period
        end
    end
    return closest, ϵ
end

function atom_nucleus(
        c :: Complex{R},
        period :: I; 
        ϵ :: R = eps(zero(R))
    ) :: Complex{R} where {R <: Real, I <: Integer}

    # newton's method step loop
    ϵ² = ϵ^2
    ϵᵢ² = 2*ϵ²
    while true
        i = zero(I)
        ∂c = zero(Complex{R})
        z = zero(Complex{R})
        # mandelbrot iteration loop
        while i < period
            ∂c = 2*z*∂c + 1
            z = z*z + c
            i += 1
        end
        ϵᵢ² = real(z)^2 + imag(z)^2
        
        if (ϵᵢ² < ϵ²)
            return c
        end
        # newton step
        c = c - z / ∂c
    end
end

function normalize_escape(iteration :: Integer, magnitude² :: Real, max_iteration :: Integer)
    if iteration >= max_iteration
        return zero(typeof(magnitude²))
    end
    (iteration + 1) - log2(log(magnitude²) / 2)
end

function overlay_debug_colors(iteration :: I, pixel :: RGB, max_iteration :: I) :: RGB where {I <: Integer}
    if iteration == max_iteration + 1
        return RGB(0.0, 1.0, 0.0)
    end
    if iteration == max_iteration + 2
        return RGB(0.0, 0.0, 1.0)
    end
    return pixel
end

function debug_coloring(μ :: Matrix{R}, iteration :: Matrix{I}, magnitude² :: Matrix{R}, max_iteration :: I) :: Matrix{RGB} where {R <: Real, I <: Integer}
    image = cosine_coloring.(μ, 1.0, 1.0, 1.0)
    return overlay_debug_colors.(iteration, image, max_iteration)
end

function period_coloring(
        periods :: Matrix{I}, 
        magnitudes :: Matrix{R}
    ) :: Matrix{RGB} where {R <: Real, I <: Integer}

    inf_ind = map(magnitudes) do mag
        isnan(mag) || isinf(mag)
    end
    magnitudes[inf_ind] .= 0.0

    m = minimum(magnitudes)
    M = maximum(magnitudes)

    magnitudes_normalized = 1.0 .- (magnitudes .- m) ./ (M - m)
    img = map(cosine_coloring.(Float64.(periods), cosine_scheme...), magnitudes_normalized) do pixel, mag
        pixel .* mag
    end

    #img[inf_ind] .= RGB(1.0, 1.0, 1.0)
    return img
end

function trimmed_mean_downsample(m :: Matrix{R}, K :: I) :: Matrix{R} where {R <: Real, I <: Integer}
    X, Y = size(m)
    newX = I(X / K)
    newY = I(Y / K)

    ds = Matrix{R}(undef, (newX, newY))

    for x = 1:newX
        for y = 1:newY
            region = @view m[1 + K*(x-1) : x*K, 1 + K*(y-1) : y*K]
            ds[x, y] = mean(sort(region[:])[2:end-1])
        end
    end

    return ds
end

function gaussian_blur(m :: Matrix{<:Real}, σ :: Real)
    kernel = Kernel.gaussian(σ)
    imfilter(m, kernel)
end

function render_pipeline(
        center :: Complex{BigFloat}, 
        zoom :: R, resolution :: Tuple{I, I}, 
        super_sampling :: I, max_iteration :: I, escape_radius :: R; 
        gpu :: Bool = true,
        renderer :: Function = BLA_renderer,
        reference_point :: Complex{BigFloat} = renderer ∈ perturbation_renderers ? center : zero(BigFloat{datatype}),
        datatype :: DataType = Float64
    ) where {R <: Real, I <: Integer}

    # set up complex plane mesh
    Δ = Complex{datatype}(center - reference_point)
    super_resolution = super_sampling .* resolution
    δ = δ_mesh(zoom, super_resolution, datatype) .+ Δ
    
    pixel_data = (δ, )
    reference_data = ()
    scalar_data = (max_iteration, escape_radius)
    
    # create reference orbit
    if renderer ∈ perturbation_renderers
        println("Calculating reference orbit...")
        ϵ = datatype(zoom) / minimum(resolution) # pixel tolerance
        @time reference = reference_orbit(reference_point, max_iteration, datatype, ϵ)
        reference_data = (reference_data..., reference)
    end
    
    # create BLA table
    if renderer ∈ BLA_renderers
        println("Creating BLA table...")
        # generate bilinear aproximation table
        @time BLA_table = create_BLA_table(reference, δ)
        # add bla table to the reference data
        reference_data = (reference_data..., BLA_table...)
    end

    # per pixel iteration
    println("Iterating pixels...")
    println("   super sampling dimensions: $super_resolution")
    @time result = render(renderer, pixel_data, reference_data, scalar_data; gpu)
    
    iterations = first.(result)
    magnitudes = Float64.(last.(result))

    # process result into image
    μ = normalize_escape.(iterations, magnitudes, max_iteration)

    if super_sampling != 1 # anti-alias and downsamle
        μ = trimmed_mean_downsample(μ, super_sampling)
    end

    return μ
end

#end
