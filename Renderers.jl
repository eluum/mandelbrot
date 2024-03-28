# Renderers.jl contains functions for rendering individual pixels

function render(
        renderer :: Function, 
        pixel_data :: Tuple, 
        reference_data :: Tuple, 
        scalar_data :: Tuple; 
        gpu :: Bool = true)

    if gpu # parallelize on the gpu
        # upload pixel and reference data to gpu
        pixel_gpu = CuArray.(pixel_data)
        reference_gpu = CuArray.(reference_data)
        result = Matrix(renderer.(pixel_gpu..., Ref.(reference_gpu)..., scalar_data...))
    else # parallelize on the cpu
        δ_mesh = pixel_data[1]
        
        # pre allocate space for returned value
        return_type = Base.return_types(renderer)[1]
        result = Matrix{return_type}(undef, size(δ_mesh)...)
        
        Threads.@threads for index in 1:length(δ_mesh)
            @inbounds result[index] = renderer(getindex.(pixel_data, index)..., reference_data..., scalar_data...)
        end
    end
    return result
end

function naive_renderer(
        c :: Complex{R}, 
        max_iteration :: I,
        escape_radius :: R
    ) :: Tuple{I, R} where {R <: Real, I <: Integer}

    iteration = zero(I)
    magnitude² = zero(R)
    z = zero(R)

    while iteration < max_iteration
        z = z*z + c
        iteration += one(I)        
        # check escape
        magnitude² = real(z)^2 + imag(z)^2
        if magnitude² > escape_radius
            break
        end
    end
    return iteration, magnitude²
end

function naive_interior_renderer(
        c :: Complex{R},
        max_iteration :: I,
        escape_radius :: R;
        ϵ :: R = R(0.001^2)
    ) :: Tuple{I, R} where {R <: Real, I <: Integer}

    iteration = zero(I)
    magnitude² = zero(R)
    z = zero(R)
    ∂z = one(Complex{R})

    while iteration < max_iteration
        z = z*z + c
        ∂z = R(2)*z*∂z
        iteration += one(I)        

        # check escape
        magnitude² = real(z)^2 + imag(z)^2
        if magnitude² > escape_radius
            break
        end
        
        if real(∂z)^2 + imag(∂z)^2 < ϵ
            iteration = max_iteration
            break
        end
    end
    return iteration, magnitude²
end

function perturbation_renderer(
        δ₀ :: Complex{R}, 
        reference :: AbstractVector{Complex{R}}, 
        max_iteration :: I,
        escape_radius :: R
    ) :: Tuple{I, R} where {R <: Real, I <:Integer}

    iteration = zero(I)
    reference_iteration = one(I)
    reference_length = I(length(reference))

    magnitude² = zero(R)
    δ_magnitude² = zero(R)

    Z = zero(Complex{R})
    δ = zero(Complex{R})

    while iteration < max_iteration

        δ = I(2)*Z*δ + δ*δ + δ₀

        @inbounds Z = reference[reference_iteration]
        reference_iteration += one(I) # next reference iteration

        z = Z + δ
        iteration += one(I)

        # check escape
        magnitude² = real(z)^2 + imag(z)^2
        if magnitude² > escape_radius
            break
        end

        # check reference
        δ_magnitude² = real(δ)^2 + imag(δ)^2
        if magnitude² < δ_magnitude² || reference_iteration > reference_length # do the rebase
            δ = z
            Z = zero(Complex{R})
            reference_iteration = one(I)
        end
    end
    return iteration, magnitude²
end

function perturbation_renderer(
        δ₀ :: Complex{R},
        reference :: AbstractVector{Complex{R}},
        max_iteration :: I,
        escape_radius :: R
    ) :: Tuple{I, R} where {R <: ExtendedRangeFloat, I <:Integer}

    iteration = zero(I)
    reference_iteration = one(I)
    reference_length = I(length(reference))

    magnitude² = zero(R)
    δ_magnitude² = zero(R)

    Z = zero(Complex{R})
    δ = zero(Complex{R})

    while iteration < max_iteration

        δ = I(2)*Z*δ + δ*δ + δ₀
        δ = normalize(δ)

        @inbounds Z = reference[reference_iteration]
        reference_iteration += I(1) # next reference iteration

        z = Z + δ
        iteration += I(1)

        # check escape
        magnitude² = real(z)^2 + imag(z)^2
        if magnitude² > escape_radius
            break
        end

        # check reference
        δ_magnitude² = real(δ)^2 + imag(δ)^2
        if magnitude² < δ_magnitude² || reference_iteration > reference_length # do the rebase
            δ = z
            Z = zero(Complex{R})
            reference_iteration = I(1)
        end
    end
    return iteration, magnitude²
end

function create_BLA_table(
        reference :: Vector{Complex{R}}, 
        δ_grid :: Matrix{Complex{R}}
    ) :: Tuple{Vector{R}, Vector{Complex{R}}, Vector{Complex{R}}} where 
        {R <: Real}

    δ_max = maximum(abs.(δ_grid))
    
    base_length = length(reference) - 1

    # create lookup tables
    l = base_length
    table_size = base_length
    table_levels = 1
    while l > 1
        l = fld(l, 2)
        table_size += l
        table_levels += 1
    end

    A = zeros(Complex{R}, table_size)
    B = ones(Complex{R}, table_size)
    radii = zeros(R, table_size)

    ϵ = eps(one(R))/2
    
    # first level
    Threads.@threads for i = 1:base_length
        A_t = 2 * reference[i]
        A[i] = A_t
        radii[i] = max(zero(R), ϵ*(abs(reference[i]) - δ_max) / (abs(A_t) + 1))
    end
    
    # remaining levels
    level_length = base_length
    current_offset = level_length
    previous_offset = 0
    for level = 1:table_levels-1
        # each level is half the length of the previous  
        level_length = fld(level_length, 2)

        # reduce level
        Threads.@threads for i = 1:level_length
            A[current_offset + i] = A[previous_offset + 2*i] * A[previous_offset + 2*i - 1]
            B[current_offset + i] = A[previous_offset + 2*i] * B[previous_offset + 2*i - 1] + B[previous_offset + 2*i]
            radii[current_offset + i] = min(
                radii[previous_offset + 2*i - 1], 
                max(zero(R), 
                   (radii[previous_offset + 2*i] - abs(B[previous_offset + 2*i-1])*δ_max) / abs(A[previous_offset + 2*i - 1]) ))
        end

        # progress to next level
        previous_offset = current_offset
        current_offset += level_length
    end

    # precompute preskip
    #level_offset = 0
    #level_length = ref_length
    #level = -1
    #while level <= table_levels
    #    if δ_max >= radii[level_offset + 1]
    #        break
    #    end
    #    level_offset += level_length
    #    level_length = cld(level_length, 2)
    #    level += 1
    #end
    #if level < 0
    #    preskip = 0
    #else
    #    preskip = 2^level
    #end
    #println(preskip)

    radii = radii.^2 # saves per pixel per iteration square root later
    return radii, A, B # finished tables
end

function create_BLA_table(
        reference :: Vector{Complex{R}},
        δ_grid :: Matrix{Complex{R}}
    ) :: Tuple{Vector{R}, Vector{Complex{R}}, Vector{Complex{R}}} where 
        {R <: ExtendedRangeFloat}
    
    δ_max = maximum(normalize.(abs.(δ_grid)))
    
    base_length = length(reference) - 1

    # create lookup tables
    l = base_length
    table_size = base_length
    table_levels = 1
    while l > 1
        l = fld(l, 2)
        table_size += l
        table_levels += 1
    end

    A = zeros(Complex{R}, table_size)
    B = ones(Complex{R}, table_size)
    radii = zeros(R, table_size)

    ϵ = eps(one(R))/2
    
    # first level
    Threads.@threads for i = 1:base_length
        A_t = normalize(2 * reference[i])
        A[i] = A_t
        radii[i] = normalize(max(zero(R), ϵ*(abs(reference[i]) - δ_max) / (abs(A_t) + 1)))
    end
    
    # remaining levels
    level_length = base_length
    current_offset = level_length
    previous_offset = 0
    for level = 1:table_levels-1
        # each level is half the length of the previous  
        level_length = fld(level_length, 2)

        # reduce level
        Threads.@threads for i = 1:level_length
            A[current_offset + i] = normalize(A[previous_offset + 2*i] * A[previous_offset + 2*i - 1])
            B[current_offset + i] = normalize(A[previous_offset + 2*i] * B[previous_offset + 2*i - 1] + B[previous_offset + 2*i])
            radii[current_offset + i] = normalize(min(
                radii[previous_offset + 2*i - 1], 
                max(zero(R), 
                   (radii[previous_offset + 2*i] - abs(B[previous_offset + 2*i-1])*δ_max) / abs(A[previous_offset + 2*i - 1]) )))
        end

        # progress to next level
        previous_offset = current_offset
        current_offset += level_length
    end

    # precompute preskip
    #level_offset = 0
    #level_length = ref_length
    #level = -1
    #while level <= table_levels
    #    if δ_max >= radii[level_offset + 1]
    #        break
    #    end
    #    level_offset += level_length
    #    level_length = cld(level_length, 2)
    #    level += 1
    #end
    #if level < 0
    #    preskip = 0
    #else
    #    preskip = 2^level
    #end
    #println(preskip)

    radii = normalize.(radii.^2) # saves per pixel per iteration square root later
    return radii, A, B # finished tables
end

function BLA_renderer(
        δ₀ :: Complex{R},  
        reference :: AbstractVector{Complex{R}}, 
        radii :: AbstractVector{R}, 
        A :: AbstractVector{Complex{R}}, 
        B :: AbstractVector{Complex{R}}, 
        max_iteration :: I,
        escape_radius :: R
    ) :: Tuple{I, R} where {R <: Real, I <: Integer}
    
    iteration = zero(I)
    reference_iteration = one(I)
    best = one(I)
    reference_length :: I = I(length(reference))
    base_length :: I = reference_length - 1
    table_length :: I = I(length(radii))
    skipped = zero(I)

    magnitude² = zero(R)
    δ_magnitude² = zero(R)

    δ = δ₀
    δ_magnitude² = real(δ)^2 + imag(δ)^2

    while iteration < max_iteration
        
        # first linear approx
        if δ_magnitude² < radii[reference_iteration]
            
            # find largest allowable skip
            table_index = reference_iteration
            best = table_index
            skipped = one(I)
            
            # check if 2*current skip is allowed
            if Bool(reference_iteration % I(2)) && reference_iteration < base_length
                # offset to next table level
                level_offset = base_length
                
                # next level values
                level_length = fld(base_length, I(2))
                level_index = cld(reference_iteration, I(2))
                table_index = level_offset + level_index
                while table_index <= table_length && (δ_magnitude² < radii[table_index] && Bool(level_index % I(2)) && level_index < level_length)
                    skipped *= I(2)
                    best = table_index # note best encountered skip

                    # offset to next table level
                    level_offset += level_length
                    
                    # next level values
                    level_length = fld(level_length, I(2))
                    level_index = cld(level_index, I(2))
                    table_index = level_offset + level_index
                end
            end

            # apply largest allowed skip
            @inbounds δ = A[best]*δ + B[best]*δ₀
            reference_iteration += skipped # next reference iteration
            iteration += skipped
        else
            # fallback to single step quadratic iteration
            @inbounds δ = A[reference_iteration]*δ + δ^2 + δ₀ 
            reference_iteration += one(I)
            iteration += one(I)
        end
        
        # advance to appropriate reference iteration
        @inbounds Z = reference[reference_iteration]
        z = Z + δ

        # check escape
        magnitude² = real(z)^2 + imag(z)^2
        if magnitude² > escape_radius
            break
        end
        
        # check reference rebase condition
        δ_magnitude² = real(δ)^2 + imag(δ)^2
        if magnitude² < δ_magnitude² || reference_iteration > base_length
            δ = z*z + δ₀
            δ_magnitude² = real(δ)^2 + imag(δ)^2
            reference_iteration = I(1)
        end
    end
    return iteration, magnitude²
end

function BLA_renderer(
        δ₀ :: Complex{R},  
        reference :: AbstractVector{Complex{R}}, 
        radii :: AbstractVector{R}, 
        A :: AbstractVector{Complex{R}}, 
        B :: AbstractVector{Complex{R}}, 
        max_iteration :: I,
        escape_radius :: R
    ) :: Tuple{I, R} where {R <: ExtendedRangeFloat, I <: Integer}
    
    iteration = zero(I)
    reference_iteration = one(I)
    best = one(I)
    reference_length :: I = I(length(reference))
    base_length :: I = reference_length - 1
    table_length :: I = I(length(radii))
    skipped = zero(I)

    magnitude² = zero(R)
    δ_magnitude² = zero(R)

    δ = δ₀
    δ_magnitude² = real(δ)^2 + imag(δ)^2

    while iteration < max_iteration
        
        # first linear approx
        if δ_magnitude² < radii[reference_iteration]
            
            # find largest allowable skip
            table_index = reference_iteration
            best = table_index
            skipped = one(I)
            
            # check if 2*current skip is allowed
            if Bool(reference_iteration % I(2)) && reference_iteration < base_length
                # offset to next table level
                level_offset = base_length
                
                # next level values
                level_length = fld(base_length, I(2))
                level_index = cld(reference_iteration, I(2))
                table_index = level_offset + level_index
                while table_index <= table_length && (δ_magnitude² < radii[table_index] && Bool(level_index % I(2)) && level_index < level_length)
                    skipped *= I(2)
                    best = table_index # note best encountered skip

                    # offset to next table level
                    level_offset += level_length
                    
                    # next level values
                    level_length = fld(level_length, I(2))
                    level_index = cld(level_index, I(2))
                    table_index = level_offset + level_index
                end
            end

            # apply largest allowed skip
            @inbounds δ = A[best]*δ + B[best]*δ₀
            reference_iteration += skipped # next reference iteration
            iteration += skipped
        else
            # fallback to single step quadratic iteration
            @inbounds δ = A[reference_iteration]*δ + δ*δ + δ₀ 
            reference_iteration += one(I)
            iteration += one(I)
        end
        
        # keep size in bounds
        δ = normalize(δ)
        
        # advance to appropriate reference iteration
        @inbounds Z = reference[reference_iteration]
        z = Z + δ

        # check escape
        magnitude² = real(z)^2 + imag(z)^2
        if magnitude² > escape_radius
            break
        end
        
        # check reference rebase condition
        δ_magnitude² = real(δ)^2 + imag(δ)^2
        if magnitude² < δ_magnitude² || reference_iteration > base_length
            δ = normalize(z*z + δ₀)
            δ_magnitude² = real(δ)^2 + imag(δ)^2
            reference_iteration = I(1)
        end
    end
    return iteration, magnitude²
end

function perturbation_renderer_distance(
        δ₀ :: Complex{Float64},
        reference :: AbstractVector{Complex{Float64}},
        max_iteration :: Int64,
        escape_radius :: Float64,
        tolerance :: Float64 = eps(Float64)
    ) :: Tuple{Int64, Float64}

    iteration :: Int64 = 1
    magnitude² :: Float64 = 0.0

    @inbounds Z = reference[1]
    δ = δ₀
    ∂c = one(Complex{Float64})
    z = Z + δ
    while iteration < max_iteration
        ∂c = 2*z*∂c + 1
        δ = 2*Z*δ + δ^2 + δ₀
        @inbounds Z = reference[iteration + 1]

        z = Z + δ
        iteration += 1

        # check escape
        magnitude² = real(z)^2 + imag(z)^2
        if magnitude² > escape_radius
            break
        end
    end

    if iteration < max_iteration
        # additional iteration for estimating distance
        mag² = magnitude²

        extra_iteration = iteration
        while extra_iteration < max_iteration
            ∂c = 2*z*∂c + 1
            δ = 2*Z*δ + δ^2 + δ₀
            @inbounds Z = reference[extra_iteration + 1]
            z = Z + δ
            mag² = real(z)^2 + imag(z)^2
            if mag² > 1e6
                break
            end
        end

        # distance estimate
        zabs = sqrt(mag²)
        b = zabs*log(zabs) / (2*abs(∂c))

        if b < tolerance
            iteration = max_iteration
        end
    end

    return iteration, magnitude²
end

function perturbation_renderer_roots(
        δ₀ :: Complex{Float64},
        reference :: AbstractVector{Complex{Float64}}, 
        max_iteration :: Int64, 
        escape_radius :: Float64
    ) :: Tuple{Int64, Float64}

    iteration :: Int64 = 1
    magnitude² :: Float64 = 0.0

    @inbounds Z = reference[1]
    δ = δ₀

    min_mag = 4.0
    min_iter = 0

    while iteration < max_iteration

        δ = 2*Z*δ + δ^2 + δ₀
        @inbounds Z = reference[iteration + 1]

        z = Z + δ
        iteration += 1

        # check escape
        magnitude² = real(z)^2 + imag(z)^2
        if magnitude² > escape_radius
            break
        end

        # close to a root at this iteration?
        if magnitude² < min_mag
            min_mag = magnitude²
            min_iter = iteration
        end

    end
    return min_iter, min_mag
end

function perturbation_renderer_slope(
        δ₀ :: Complex{Float64}, 
        reference :: AbstractVector{Complex{Float64}}, 
        max_iteration :: Int64,
        escape_radius :: Float64
    ) :: Tuple{Int64, Float64}

    iteration :: Int64 = 1
    magnitude² :: Float64 = 0.0

    @inbounds Z = reference[1]
    δ = δ₀

    max_slope = 0.0
    max_iter = 0

    ∂c = one(Complex{Float64})
    z = Z + δ
    while iteration < max_iteration

        ∂c = 2*z*∂c + 1
        δ = 2*Z*δ + δ^2 + δ₀
        @inbounds Z 

        z = Z + δ
        iteration += 1

        # check escape
        magnitude² = real(z)^2 + imag(z)^2
        if magnitude² > escape_radius
            break
        end

        ∂c² = real(∂c)^2 + imag(∂c)^2

        # maximum slope?
        if  ∂c² > max_slope
            max_slope = ∂c²
            max_iter = iteration
        end

    end
    return max_iter, log10(max_slope)
end

# information about the various renderers 
const perturbation_renderers :: Set{Function} = Set([perturbation_renderer, BLA_renderer])
const BLA_renderers :: Set{Function} = Set([BLA_renderer])