module ExtendedRangeFloats
export ExtendedRangeFloat, normalize

struct ExtendedRangeFloat{F <: AbstractFloat, I <: Integer} <: Real
    mantissa :: F
    exponent :: I
end

# enable gpu use
import Adapt
Adapt.@adapt_structure ExtendedRangeFloat

function ExtendedRangeFloat{F, I}(float :: E) where {F <: AbstractFloat, I <: Integer, E <: ExtendedRangeFloat}
    ExtendedRangeFloat{F, I}(F(float.mantissa), I(float.exponent))
end

# construct from floating point value
function ExtendedRangeFloat{F, I}(float :: AbstractFloat) where {F <: AbstractFloat, I <: Integer}
    (mantissa, exponent) = frexp(float)
    ExtendedRangeFloat(F(mantissa), I(exponent))
end

function ExtendedRangeFloat{F, I}(x :: Real) where {F <: AbstractFloat, I <: Integer}
    ExtendedRangeFloat{F, I}(F(x))
end

function normalize(float :: ExtendedRangeFloat{F, I}) :: ExtendedRangeFloat{F, I} where {F <: AbstractFloat, I <: Integer}
    m, e = frexp(float.mantissa)

    ExtendedRangeFloat(
        m, # new mantissa
        float.exponent + I(e)) # modified exponent
end

function normalize(c :: Complex{ExtendedRangeFloat{F, I}}) :: Complex{ExtendedRangeFloat{F, I}} where {F <: AbstractFloat, I <: Integer}
    complex(normalize(real(c)), normalize(imag(c)))
end

# allows regular float types to be normalized
function normalize(float :: AbstractFloat)
    float
end

# arithematic 
import Base.+
function (+)(left :: ExtendedRangeFloat, right :: ExtendedRangeFloat) :: ExtendedRangeFloat
    if iszero(left)
        return right
    end
    if iszero(right)
        return left
    end
    e = left.exponent - right.exponent
    if e > 0
        ExtendedRangeFloat(left.mantissa + ldexp(right.mantissa, -e), left.exponent)
    elseif e < 0
        ExtendedRangeFloat(ldexp(left.mantissa, e) + right.mantissa, right.exponent)
    else # equal exponent
        ExtendedRangeFloat(left.mantissa + right.mantissa, left.exponent)
    end
end

import Base.-
function (-)(left :: ExtendedRangeFloat, right :: ExtendedRangeFloat) :: ExtendedRangeFloat
    if iszero(left)
        return -right
    end
    if iszero(right)
        return left
    end
    e = left.exponent - right.exponent
    if e > 0
        ExtendedRangeFloat(left.mantissa - ldexp(right.mantissa, -e), left.exponent)
    elseif e < 0
        ExtendedRangeFloat(ldexp(left.mantissa, e) - right.mantissa, right.exponent)
    else # equal exponent
        ExtendedRangeFloat(left.mantissa - right.mantissa, left.exponent)
    end
end

import Base.*
function (*)(left :: ExtendedRangeFloat, right :: ExtendedRangeFloat) :: ExtendedRangeFloat
    ExtendedRangeFloat(
        left.mantissa * right.mantissa,
        left.exponent + right.exponent)
end

function (-)(x :: ExtendedRangeFloat)
    ExtendedRangeFloat(-x.mantissa, x.exponent)
end

import Base.^
function (^)(base :: ExtendedRangeFloat{F, I}, exponent :: Real) :: ExtendedRangeFloat{F, I} where {F <: AbstractFloat, I <: Integer}
    ExtendedRangeFloat{F, I}(base.mantissa ^ exponent, base.exponent * I(exponent))
end

function (^)(base :: ExtendedRangeFloat{F, I}, exponent :: Integer) :: ExtendedRangeFloat where {F <: AbstractFloat, I <: Integer}
    ExtendedRangeFloat{F, I}(base.mantissa ^ exponent, base.exponent * I(exponent))
end

import Base./
function (/)(left :: ExtendedRangeFloat, right :: ExtendedRangeFloat) :: ExtendedRangeFloat
    ExtendedRangeFloat(
        left.mantissa / right.mantissa,
        left.exponent - right.exponent)
end

import Base.BigFloat
function BigFloat(x :: ExtendedRangeFloat) :: BigFloat
    ldexp(BigFloat(x.mantissa), x.exponent)
end

import Base.signbit
function signbit(x :: ExtendedRangeFloat)
    signbit(x.mantissa)
end

import Base.sign
function sign(x :: ExtendedRangeFloat)
    sign(x.mantissa)
end

import Base.sqrt
function sqrt(x :: ExtendedRangeFloat{F, I}) :: ExtendedRangeFloat{F, I} where {F <: AbstractFloat, I <: Integer}
    (q, r) = divrem(x.exponent, I(2))
    if r > 0
        return ExtendedRangeFloat{F, I}(sqrt(x.mantissa) * sqrt(F(2)), q)
    end

    if r < 0
        return ExtendedRangeFloat{F, I}(sqrt(x.mantissa) / sqrt(F(2)), q)
    end

    # no remainder
    return ExtendedRangeFloat{F, I}(sqrt(x.mantissa), q)
end

import Base.abs
function abs(x :: Complex{ExtendedRangeFloat{F, I}}) :: ExtendedRangeFloat{F, I} where {F <: AbstractFloat, I <: Integer}
    sqrt(real(x)^2 + imag(x)^2)
end

# conversions

import Base.Float64
function Float64(x :: ExtendedRangeFloat) :: Float64
    ldexp(Float64(x.mantissa), x.exponent)
end

import Base.Float32
function Float32(x :: ExtendedRangeFloat) :: Float32
    ldexp(Float32(x.mantissa), x.exponent)
end

import Base.iszero
function iszero(x :: ExtendedRangeFloat) :: Bool
    iszero(x.mantissa)
end

# comparisons

import Base.convert
function convert(::ExtendedRangeFloat{F, I}, x :: Real) where {F <: AbstractFloat, I <: Integer}
    ExtendedRangeFloat{F, I}(x)
end

import Base.promote_rule
function promote_rule(::Type{ExtendedRangeFloat{F, I}}, ::Type{R}) :: Type where {F <: AbstractFloat, I <: Integer, R <: Real}
    ExtendedRangeFloat{F, I}
end
function promote_rule(::Type{BigFloat}, ::Type{ExtendedRangeFloat{F, I}}) :: Type where {F <: AbstractFloat, I <: Integer}
    ExtendedRangeFloat{F, I}
end

import Base.<
function (<)(left :: ExtendedRangeFloat, right :: ExtendedRangeFloat) :: Bool
    sign(left - right) < 0
end

#function (<)(left :: ExtendedRangeFloat, right :: ExtendedRangeFloat) :: Bool
#    left_negative = signbit(left)
#    right_negative = signbit(right)
#    if left_negative
#        if right_negative # both negative, reverse comparison
#            if left.exponent < right.exponent
#                return false
#            end
#            if left.exponent > right.exponent
#                return true
#            end
#        else # negative < positive
#            return true
#        end
#    else # left is positive
#        if right_negative # positive ≮ negative
#            return false
#        end 
#        # both are positive
#        if left.exponent < right.exponent
#            return true
#        end
#        if left.exponent > right.exponent
#            return false
#        end
#    end
#    # if we reach here, exponents are equal
#    left.mantissa < right.mantissa
#end

import Base.==
function (==)(left :: ExtendedRangeFloat, right :: ExtendedRangeFloat) :: Bool
    normalize(left) == normalize(right)
end

import Base.eps
function eps(x :: ExtendedRangeFloat{F, I}) :: ExtendedRangeFloat{F, I} where {F <: AbstractFloat, I <: Integer}
    eps(x.mantissa)
end

end
