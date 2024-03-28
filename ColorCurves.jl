using Plots
function CosineColorCurve(R :: Integer, G :: Integer, B :: Integer) :: Function
    
    simplify = gcd(R, G, B)
    R = div(R, simplify)
    G = div(G, simplify)
    B = div(B, simplify)

    # color frequencies
    Rω = R * 2π
    Gω = G * 2π
    Bω = B * 2π
    
    function curve(μ :: Real, period :: Real = 1.0) :: RGB
        μf = Float64(μ)
        RGB(
            (1.0 - cos(Rω/period * μf)) / 2.0,
            (1.0 - cos(Gω/period * μf)) / 2.0,
            (1.0 - cos(Bω/period * μf)) / 2.0)
    end
    return curve
end

function CircleColorCurve(α :: Real, r :: Real = 1.0)

    # orthogonal unit vectors
    i = (0.0, 1.0, -1.0) ./ sqrt(2.0)
    j = (-2.0, 1.0, 1.0) ./ sqrt(6.0)

    # ensure we dont leave the bounds of RGB cube
    @assert 0.0 <= α <= 1.0 "α must be between 0.0 and 1.0"
    @assert 0.0 <= r <= 1.0 "Radius must be between 0.0 and 1.0"
    R = min(1 - α, α)

    # circular curve with central axis pointing from (0,0,0) to (1,1,1)
    function curve(μ :: Real, period :: Real = 1.0) :: RGB
        RGB((α .* (1.0, 1.0, 1.0) .+ (r*R) .* (i .* cos(2π*μ/period) .+ j .* sin(2π*μ/period)))...)
    end
end

function SpiralColorCurve(α :: Real, r :: Real = 1.0)

    # orthogonal unit vectors
    i = (0.0, 1.0, -1.0) ./ sqrt(2.0)
    j = (-2.0, 1.0, 1.0) ./ sqrt(6.0)

    # ensure we dont leave the bounds of RGB cube
    @assert 0.0 <= α <= 1.0 "α must be between 0.0 and 1.0"
    @assert 0.0 <= r <= 1.0 "Radius must be between 0.0 and 1.0"
    R = min(1 - α, α)

    # circular curve with central axis pointing from (0,0,0) to (1,1,1)
    function curve(μ :: Real, period :: Real = 1.0) :: RGB
        RGB((α .* (1.0, 1.0, 1.0) .+ (r*R) .* (i .* cos(2π*μ/period) .+ j .* sin(2π*μ/period)))...)
    end
end

function plot_color_curve(curve :: Function, N = 1000)
    t = range(0.0, 1.0, N)
    colors = curve.(t)
    R = red.(colors)
    G = green.(colors) 
    B = blue.(colors)
    plot(t, t, t, linecolor = map((x)->RGB((x .* (1,1,1))...), t),
        camera = (45, 45),
        background_color = :white, 
        gridalpha = 0.3)
    plot!(R, G, B, linecolor = colors, lw = 2.0)
end

# some cosine color schemes
default_cosine_curve = CosineColorCurve(25, 8, 12)
greyscale_cosine_curve = CosineColorCurve(1, 1, 1)
