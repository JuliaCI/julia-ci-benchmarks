# The program the inference benchmarks infer: numbers with promotion, tuples, lists, lazy
# iterators, a search tree, sorting, printing, a lazy broadcast and a random number
# generator, written against Core's builtins and intrinsics alone. No method of Base is
# reachable from it, so what the benchmarks infer changes only when this file does.
#
# Syntax that lowers to a call of a Base function is either left out (keyword arguments,
# `x^2`, dot broadcasting, comprehensions, string interpolation, destructuring, indexing a
# tuple) or dispatches to a method defined here: field access, `for` and indexing call
# Base's `getproperty`, `setproperty!`, `iterate` and `getindex`, which get methods for
# the types below. A struct is built from values of its field types, since any other
# value goes through Base's `convert`. `benchmarks/inference.jl` checks that nothing
# outside this module and Core gets inferred.
baremodule InferenceWorkload
import Base
import Base: getproperty, setproperty!, iterate, getindex

const add_int = Core.Intrinsics.add_int
const sub_int = Core.Intrinsics.sub_int
const mul_int = Core.Intrinsics.mul_int
const neg_int = Core.Intrinsics.neg_int
const checked_sdiv_int = Core.Intrinsics.checked_sdiv_int
const checked_srem_int = Core.Intrinsics.checked_srem_int
const checked_udiv_int = Core.Intrinsics.checked_udiv_int
const checked_urem_int = Core.Intrinsics.checked_urem_int
const slt_int = Core.Intrinsics.slt_int
const sle_int = Core.Intrinsics.sle_int
const ult_int = Core.Intrinsics.ult_int
const ule_int = Core.Intrinsics.ule_int
const eq_int = Core.Intrinsics.eq_int
const and_int = Core.Intrinsics.and_int
const or_int = Core.Intrinsics.or_int
const xor_int = Core.Intrinsics.xor_int
const not_int = Core.Intrinsics.not_int
const shl_int = Core.Intrinsics.shl_int
const lshr_int = Core.Intrinsics.lshr_int
const flipsign_int = Core.Intrinsics.flipsign_int
const zext_int = Core.Intrinsics.zext_int
const trunc_int = Core.Intrinsics.trunc_int
const bitcast = Core.Intrinsics.bitcast
const sitofp = Core.Intrinsics.sitofp
const uitofp = Core.Intrinsics.uitofp
const fptosi = Core.Intrinsics.fptosi
const add_float = Core.Intrinsics.add_float
const sub_float = Core.Intrinsics.sub_float
const mul_float = Core.Intrinsics.mul_float
const div_float = Core.Intrinsics.div_float
const neg_float = Core.Intrinsics.neg_float
const lt_float = Core.Intrinsics.lt_float
const le_float = Core.Intrinsics.le_float
const eq_float = Core.Intrinsics.eq_float
const abs_float = Core.Intrinsics.abs_float
const sqrt_llvm = Core.Intrinsics.sqrt_llvm
const rint_llvm = Core.Intrinsics.rint_llvm
const muladd_float = Core.Intrinsics.muladd_float

# The supertype of the structs below that are not numbers
abstract type Obj end

struct WorkloadError <: Exception
    what::Symbol
end
error(what::Symbol) = throw(WorkloadError(what))
# `@new(T(args...))` builds a `T` from values of its field types, as `T(args...)` does, but
# without the call of Base's `convert` that a constructor makes for a value that inference
# does not know to be of the field's type. The lists and the tree are built with it.
macro new(ex)
    return Base.esc(Expr(:new, ex.args...))
end

## Booleans, comparison

!(x::Bool) = not_int(x)
(&)(x::Bool, y::Bool) = and_int(x, y)
(|)(x::Bool, y::Bool) = or_int(x, y)
!==(x, y) = !(x === y)
==(x, y) = x === y
!=(x, y) = !(x == y)
>(x, y) = y < x
<=(x, y) = !(y < x)
>=(x, y) = y <= x
isless(x, y) = x < y
min(x, y) = y < x ? y : x
max(x, y) = y < x ? x : y
identity(x) = x
# `a + b + c` is one call
+(a, b, c, rest...) = +(a + b, c, rest...)
*(a, b, c, rest...) = *(a * b, c, rest...)

struct Fix1{F,T} <: Obj
    f::F
    x::T
end
(f::Fix1)(y) = f.f(f.x, y)
struct Fix2{F,T} <: Obj
    f::F
    x::T
end
(f::Fix2)(y) = f.f(y, f.x)

## Integers and floats

const BitUnsigned = Union{UInt64,UInt8}
const BitInteger = Union{Int,UInt64,UInt8}

+(x::T, y::T) where {T<:BitInteger} = add_int(x, y)
-(x::T, y::T) where {T<:BitInteger} = sub_int(x, y)
*(x::T, y::T) where {T<:BitInteger} = mul_int(x, y)
-(x::T) where {T<:BitInteger} = neg_int(x)
(&)(x::T, y::T) where {T<:BitInteger} = and_int(x, y)
(|)(x::T, y::T) where {T<:BitInteger} = or_int(x, y)
xor(x::T, y::T) where {T<:BitInteger} = xor_int(x, y)
==(x::T, y::T) where {T<:BitInteger} = eq_int(x, y)
<(x::Int, y::Int) = slt_int(x, y)
<=(x::Int, y::Int) = sle_int(x, y)
<(x::T, y::T) where {T<:BitUnsigned} = ult_int(x, y)
<=(x::T, y::T) where {T<:BitUnsigned} = ule_int(x, y)
div(x::Int, y::Int) = checked_sdiv_int(x, y)
rem(x::Int, y::Int) = checked_srem_int(x, y)
div(x::T, y::T) where {T<:BitUnsigned} = checked_udiv_int(x, y)
rem(x::T, y::T) where {T<:BitUnsigned} = checked_urem_int(x, y)
<<(x::UInt64, n::Int) = shl_int(x, n)
>>(x::UInt64, n::Int) = lshr_int(x, n)
abs(x::Int) = flipsign_int(x, x)
abs(x::BitUnsigned) = x
iseven(x::Int) = and_int(x, 1) == 0

+(x::Float64, y::Float64) = add_float(x, y)
-(x::Float64, y::Float64) = sub_float(x, y)
*(x::Float64, y::Float64) = mul_float(x, y)
/(x::Float64, y::Float64) = div_float(x, y)
-(x::Float64) = neg_float(x)
==(x::Float64, y::Float64) = eq_float(x, y)
<(x::Float64, y::Float64) = lt_float(x, y)
<=(x::Float64, y::Float64) = le_float(x, y)
abs(x::Float64) = abs_float(x)
sqrt(x::Float64) = sqrt_llvm(x)
round(x::Float64) = rint_llvm(x)
trunc(::Type{Int}, x::Float64) = fptosi(Int, x)
muladd(x::Float64, y::Float64, z::Float64) = muladd_float(x, y, z)
muladd(x, y, z) = x * y + z
/(x::Int, y::Int) = convert(Float64, x) / convert(Float64, y)

zero(::Type{Int}) = 0
zero(::Type{UInt64}) = 0x0000000000000000
zero(::Type{UInt8}) = 0x00
zero(::Type{Float64}) = 0.0
one(::Type{Int}) = 1
one(::Type{UInt64}) = 0x0000000000000001
one(::Type{UInt8}) = 0x01
one(::Type{Float64}) = 1.0
zero(x::Number) = zero(typeof(x))
one(x::Number) = one(typeof(x))
iszero(x) = x == zero(x)

## Conversion and promotion

convert(::Type{T}, x::T) where {T} = x
convert(::Type{Float64}, x::Int) = sitofp(Float64, x)
convert(::Type{Float64}, x::UInt64) = uitofp(Float64, x)
convert(::Type{Float64}, x::UInt8) = uitofp(Float64, x)
convert(::Type{Int}, x::UInt8) = zext_int(Int, x)
convert(::Type{UInt64}, x::UInt8) = zext_int(UInt64, x)
convert(::Type{UInt64}, x::Int) = x < 0 ? error(:inexact) : bitcast(UInt64, x)
convert(::Type{UInt8}, x::Int) = (0 <= x) & (x < 256) ? trunc_int(UInt8, x) : error(:inexact)
function convert(::Type{Int}, x::UInt64)
    y = bitcast(Int, x)
    return y < 0 ? error(:inexact) : y
end
function convert(::Type{Int}, x::Float64)
    y = fptosi(Int, x)
    return sitofp(Float64, y) == x ? y : error(:inexact)
end

promote_rule(::Type, ::Type) = Union{}
promote_rule(::Type{Float64}, ::Type{Int}) = Float64
promote_rule(::Type{Float64}, ::Type{UInt64}) = Float64
promote_rule(::Type{Float64}, ::Type{UInt8}) = Float64
promote_rule(::Type{Int}, ::Type{UInt8}) = Int
promote_rule(::Type{UInt64}, ::Type{UInt8}) = UInt64
promote_rule(::Type{UInt64}, ::Type{Int}) = UInt64

promote_type(::Type{Union{}}, ::Type{Union{}}) = Union{}
promote_type(::Type{T}, ::Type{T}) where {T} = T
promote_type(::Type{T}, ::Type{Union{}}) where {T} = T
promote_type(::Type{Union{}}, ::Type{T}) where {T} = T
promote_type(::Type{T}, ::Type{S}) where {T,S} = promote_result(T, S, promote_rule(T, S), promote_rule(S, T))
promote_result(::Type, ::Type, ::Type{T}, ::Type{S}) where {T,S} = promote_type(T, S)
promote_result(::Type, ::Type, ::Type{Union{}}, ::Type{Union{}}) = error(:no_promotion)

promote(x::T, y::T) where {T} = (x, y)
function promote(x, y)
    T = promote_type(typeof(x), typeof(y))
    return (convert(T, x), convert(T, y))
end

+(x::Number, y::Number) = +(promote(x, y)...)
-(x::Number, y::Number) = -(promote(x, y)...)
*(x::Number, y::Number) = *(promote(x, y)...)
/(x::Number, y::Number) = /(promote(x, y)...)
==(x::Number, y::Number) = ==(promote(x, y)...)
<(x::Number, y::Number) = <(promote(x, y)...)
+(x::T, y::T) where {T<:Number} = error(:no_method)
-(x::T, y::T) where {T<:Number} = error(:no_method)
*(x::T, y::T) where {T<:Number} = error(:no_method)
/(x::T, y::T) where {T<:Number} = error(:no_method)
<(x::T, y::T) where {T<:Number} = error(:no_method)
==(x::T, y::T) where {T<:Number} = x === y

## Tuples

struct Val{N} end
argtail(x, rest...) = rest
tail(t::Tuple) = argtail(t...)
first(t::Tuple) = getfield(t, 1)
length(t::Tuple) = nfields(t)
map(f, t::Tuple{}) = ()
map(f, t::Tuple{Any}) = (f(getfield(t, 1)),)
map(f, t::Tuple{Any,Any}) = (f(getfield(t, 1)), f(getfield(t, 2)))
map(f, t::Tuple) = (f(getfield(t, 1)), map(f, tail(t))...)
map(f, t::Tuple{}, s::Tuple{}) = ()
map(f, t::Tuple, s::Tuple) = (f(first(t), first(s)), map(f, tail(t), tail(s))...)
foldl(op, acc, t::Tuple{}) = acc
foldl(op, acc, t::Tuple) = foldl(op, op(acc, first(t)), tail(t))
all(f, t::Tuple{}) = true
all(f, t::Tuple) = f(first(t)) && all(f, tail(t))
ntuple(f, ::Val{0}) = ()
ntuple(f, ::Val{N}) where {N} = (ntuple(f, Val{N - 1}())..., f(N))
isless(t::Tuple{}, s::Tuple{}) = false
isless(t::Tuple, s::Tuple) = isless(first(t), first(s)) || (!isless(first(s), first(t)) && isless(tail(t), tail(s)))

## Elementary functions

evalpoly(x, p::Tuple{Any}) = getfield(p, 1)
evalpoly(x, p::Tuple) = muladd(x, evalpoly(x, tail(p)), getfield(p, 1))

const INV_PI = 0.3183098861837907
const PI_HI = 3.141592653589793
const PI_LO = 1.2246467991473532e-16
const HALF_PI = 1.5707963267948966
const INV_LN2 = 1.4426950408889634
const LN2 = 0.6931471805599453

function sin(x::Float64)
    k = round(x * INV_PI)
    r = muladd(k, -PI_LO, muladd(k, -PI_HI, x))
    r2 = r * r
    s = muladd(r * r2, evalpoly(r2, (-0.16666666666666666, 0.008333333333333333, -0.0001984126984126984,
                                     2.7557319223985893e-6, -2.505210838544172e-8, 1.6059043836821613e-10)), r)
    return iseven(trunc(Int, k)) ? s : -s
end
cos(x::Float64) = sin(x + HALF_PI)
function exp(x::Float64)
    k = round(x * INV_LN2)
    r = muladd(k, -LN2, x)
    p = evalpoly(r, (1.0, 1.0, 0.5, 0.16666666666666666, 0.041666666666666664, 0.008333333333333333,
                     0.001388888888888889, 0.0001984126984126984, 2.48015873015873e-5, 2.7557319223985893e-6))
    # 2^k, from its bits
    return p * bitcast(Float64, shl_int(trunc(Int, k) + 1023, 52))
end
function log(x::Float64)
    x <= 0.0 && error(:domain)
    bits = bitcast(UInt64, x)
    e = convert(Int, bits >> 52) - 1023
    m = bitcast(Float64, (bits & 0x000fffffffffffff) | 0x3ff0000000000000)
    s = (m - 1.0) / (m + 1.0)
    s2 = s * s
    return muladd(convert(Float64, e), LN2, 2.0 * s * evalpoly(s2, (1.0, 0.3333333333333333, 0.2, 0.14285714285714285,
                                                                    0.1111111111111111, 0.09090909090909091, 0.07692307692307693)))
end
sin(x::Real) = sin(convert(Float64, x))
cos(x::Real) = cos(convert(Float64, x))
exp(x::Real) = exp(convert(Float64, x))
sqrt(x::Real) = sqrt(convert(Float64, x))

## Rational, complex and dual numbers

struct Rat{T<:Integer} <: Real
    num::T
    den::T
end
struct Cplx{T<:Real} <: Number
    re::T
    im::T
end
struct Dual{T<:Real} <: Real
    val::T
    der::T
end
Cplx(re::Real, im::Real) = Cplx(promote(re, im)...)
Dual(val::Real, der::Real) = Dual(promote(val, der)...)
getproperty(x::Union{Obj,Rat,Cplx,Dual}, f::Symbol) = getfield(x, f)
setproperty!(x::Obj, f::Symbol, v) = setfield!(x, f, v)

function gcd(a::T, b::T) where {T<:Integer}
    while !iszero(b)
        t = b
        b = rem(a, b)
        a = t
    end
    return abs(a)
end
function rational(n::T, d::T) where {T<:Integer}
    iszero(d) && error(:zero_denominator)
    g = gcd(n, d)
    n = div(n, g)
    d = div(d, g)
    return d < zero(T) ? Rat{T}(-n, -d) : Rat{T}(n, d)
end
//(n::Integer, d::Integer) = rational(promote(n, d)...)
+(x::Rat{T}, y::Rat{T}) where {T<:Integer} = rational(x.num * y.den + y.num * x.den, x.den * y.den)
-(x::Rat{T}, y::Rat{T}) where {T<:Integer} = rational(x.num * y.den - y.num * x.den, x.den * y.den)
*(x::Rat{T}, y::Rat{T}) where {T<:Integer} = rational(x.num * y.num, x.den * y.den)
/(x::Rat{T}, y::Rat{T}) where {T<:Integer} = rational(x.num * y.den, x.den * y.num)
-(x::Rat{T}) where {T<:Integer} = Rat{T}(-x.num, x.den)
==(x::Rat{T}, y::Rat{T}) where {T<:Integer} = (x.num == y.num) & (x.den == y.den)
<(x::Rat{T}, y::Rat{T}) where {T<:Integer} = x.num * y.den < y.num * x.den
abs(x::Rat{T}) where {T<:Integer} = Rat{T}(abs(x.num), x.den)
zero(::Type{Rat{T}}) where {T<:Integer} = Rat{T}(zero(T), one(T))
one(::Type{Rat{T}}) where {T<:Integer} = Rat{T}(one(T), one(T))
convert(::Type{Rat{T}}, x::Rat{T}) where {T<:Integer} = x
convert(::Type{Rat{T}}, x::Rat) where {T<:Integer} = Rat{T}(convert(T, x.num), convert(T, x.den))
convert(::Type{Rat{T}}, x::Integer) where {T<:Integer} = Rat{T}(convert(T, x), one(T))
convert(::Type{Float64}, x::Rat) = convert(Float64, x.num) / convert(Float64, x.den)
promote_rule(::Type{Rat{T}}, ::Type{S}) where {T<:Integer,S<:Integer} = Rat{promote_type(T, S)}
promote_rule(::Type{Rat{T}}, ::Type{Rat{S}}) where {T<:Integer,S<:Integer} = Rat{promote_type(T, S)}
promote_rule(::Type{Rat{T}}, ::Type{Float64}) where {T<:Integer} = Float64

+(z::Cplx{T}, w::Cplx{T}) where {T<:Real} = Cplx{T}(z.re + w.re, z.im + w.im)
-(z::Cplx{T}, w::Cplx{T}) where {T<:Real} = Cplx{T}(z.re - w.re, z.im - w.im)
*(z::Cplx{T}, w::Cplx{T}) where {T<:Real} = Cplx{T}(z.re * w.re - z.im * w.im, z.re * w.im + z.im * w.re)
function /(z::Cplx{T}, w::Cplx{T}) where {T<:Real}
    d = abs2(w)
    return Cplx((z.re * w.re + z.im * w.im) / d, (z.im * w.re - z.re * w.im) / d)
end
-(z::Cplx{T}) where {T<:Real} = Cplx{T}(-z.re, -z.im)
==(z::Cplx{T}, w::Cplx{T}) where {T<:Real} = (z.re == w.re) & (z.im == w.im)
conj(z::Cplx{T}) where {T<:Real} = Cplx{T}(z.re, -z.im)
cplx(re::Real, im::Real) = Cplx(re, im)
abs2(z::Cplx) = z.re * z.re + z.im * z.im
abs(z::Cplx) = sqrt(abs2(z))
exp(z::Cplx) = (e = exp(z.re); Cplx(e * cos(z.im), e * sin(z.im)))
zero(::Type{Cplx{T}}) where {T<:Real} = Cplx{T}(zero(T), zero(T))
one(::Type{Cplx{T}}) where {T<:Real} = Cplx{T}(one(T), zero(T))
convert(::Type{Cplx{T}}, z::Cplx{T}) where {T<:Real} = z
convert(::Type{Cplx{T}}, z::Cplx) where {T<:Real} = Cplx{T}(convert(T, z.re), convert(T, z.im))
convert(::Type{Cplx{T}}, x::Real) where {T<:Real} = Cplx{T}(convert(T, x), zero(T))
promote_rule(::Type{Cplx{T}}, ::Type{S}) where {T<:Real,S<:Real} = Cplx{promote_type(T, S)}
promote_rule(::Type{Cplx{T}}, ::Type{Cplx{S}}) where {T<:Real,S<:Real} = Cplx{promote_type(T, S)}

+(x::Dual{T}, y::Dual{T}) where {T<:Real} = Dual{T}(x.val + y.val, x.der + y.der)
-(x::Dual{T}, y::Dual{T}) where {T<:Real} = Dual{T}(x.val - y.val, x.der - y.der)
*(x::Dual{T}, y::Dual{T}) where {T<:Real} = Dual{T}(x.val * y.val, x.val * y.der + x.der * y.val)
/(x::Dual{T}, y::Dual{T}) where {T<:Real} = Dual(x.val / y.val, (x.der * y.val - x.val * y.der) / (y.val * y.val))
-(x::Dual{T}) where {T<:Real} = Dual{T}(-x.val, -x.der)
==(x::Dual{T}, y::Dual{T}) where {T<:Real} = x.val == y.val
<(x::Dual{T}, y::Dual{T}) where {T<:Real} = x.val < y.val
abs(x::Dual) = x.val < zero(x.val) ? -x : x
sin(x::Dual) = Dual(sin(x.val), cos(x.val) * x.der)
cos(x::Dual) = Dual(cos(x.val), -(sin(x.val) * x.der))
exp(x::Dual) = (e = exp(x.val); Dual(e, e * x.der))
sqrt(x::Dual) = (s = sqrt(x.val); Dual(s, x.der / (s + s)))
zero(::Type{Dual{T}}) where {T<:Real} = Dual{T}(zero(T), zero(T))
one(::Type{Dual{T}}) where {T<:Real} = Dual{T}(one(T), zero(T))
convert(::Type{Dual{T}}, x::Dual{T}) where {T<:Real} = x
convert(::Type{Dual{T}}, x::Dual) where {T<:Real} = Dual{T}(convert(T, x.val), convert(T, x.der))
convert(::Type{Dual{T}}, x::Real) where {T<:Real} = Dual{T}(convert(T, x), zero(T))
convert(::Type{Float64}, x::Dual) = convert(Float64, x.val)
promote_rule(::Type{Dual{T}}, ::Type{S}) where {T<:Real,S<:Real} = Dual{promote_type(T, S)}
promote_rule(::Type{Dual{T}}, ::Type{Dual{S}}) where {T<:Real,S<:Real} = Dual{promote_type(T, S)}
derivative(f, x::Float64) = f(Dual(x, 1.0)).der

## Ranges, pairs, lists

struct Range <: Obj
    start::Int
    stop::Int
end
(:)(a::Int, b::Int) = Range(a, b)
iterate(r::Range) = r.stop < r.start ? nothing : (r.start, r.start)
iterate(r::Range, i::Int) = i == r.stop ? nothing : (i + 1, i + 1)
length(r::Range) = max(0, r.stop - r.start + 1)
in(x::Int, r::Range) = (r.start <= x) & (x <= r.stop)
function getindex(r::Range, i::Int)
    Base.@boundscheck (1 <= i) & (i <= length(r)) || error(:bounds)
    return r.start + i - 1
end

struct Pair{A,B} <: Obj
    first::A
    second::B
end
first(p::Pair) = p.first
second(p::Pair) = p.second
isless(p::Pair, q::Pair) = isless(p.first, q.first) || (!isless(q.first, p.first) && isless(p.second, q.second))

abstract type List{T} <: Obj end
struct Nil{T} <: List{T} end
struct Cons{T} <: List{T}
    head::T
    tail::List{T}
end
isempty(l::Nil) = true
isempty(l::Cons) = false
iterate(l::List) = iterate(l, l)
iterate(::List, s::Nil) = nothing
iterate(::List, s::Cons) = (s.head, s.tail)
list(::Type{T}, t::Tuple{}) where {T} = Nil{T}()
list(::Type{T}, t::Tuple) where {T} = @new(Cons{T}(convert(T, first(t)), list(T, tail(t))))
function length(l::List)
    n = 0
    for _ in l
        n += 1
    end
    return n
end
function reverse(l::List{T}) where {T}
    acc = Nil{T}()
    for x in l
        acc = @new(Cons{T}(x, acc))
    end
    return acc
end
function getindex(l::List, i::Int)
    n = 1
    for x in l
        n == i && return x
        n += 1
    end
    error(:bounds)
end
function drop(l::List, n::Int)
    while (n > 0) & !isempty(l)
        l = l.tail
        n -= 1
    end
    return l
end
function take(l::List{T}, n::Int) where {T}
    acc = Nil{T}()
    for x in l
        n == 0 && break
        acc = @new(Cons{T}(x, acc))
        n -= 1
    end
    return reverse(acc)
end

# A mutable sequence
mutable struct Vec{T} <: Obj
    items::List{T}
    len::Int
end
vec(l::List{T}) where {T} = @new(Vec{T}(l, length(l)))
length(v::Vec) = v.len
iterate(v::Vec) = iterate(v.items)
iterate(v::Vec, s::List) = iterate(v.items, s)
function getindex(v::Vec, i::Int)
    Base.@boundscheck (1 <= i) & (i <= v.len) || error(:bounds)
    return v.items[i]
end
function push!(v::Vec{T}, x::T) where {T}
    v.items = reverse(@new(Cons{T}(x, reverse(v.items))))
    v.len += 1
    return v
end
function fill!(v::Vec{T}, x::T) where {T}
    acc = Nil{T}()
    for _ in 1:v.len
        acc = @new(Cons{T}(x, acc))
    end
    v.items = acc
    return v
end

## Lazy iterators and reductions

struct Map{F,I} <: Obj
    f::F
    itr::I
end
struct Filter{F,I} <: Obj
    pred::F
    itr::I
end
struct Zip{A,B} <: Obj
    a::A
    b::B
end
struct Enumerate{I} <: Obj
    itr::I
end
mapnext(m::Map, ::Nothing) = nothing
mapnext(m::Map, it::Tuple) = (m.f(getfield(it, 1)), getfield(it, 2))
iterate(m::Map) = mapnext(m, iterate(m.itr))
iterate(m::Map, s) = mapnext(m, iterate(m.itr, s))
function filternext(f::Filter, it)
    while it !== nothing
        f.pred(getfield(it, 1)) && return it
        it = iterate(f.itr, getfield(it, 2))
    end
    return nothing
end
iterate(f::Filter) = filternext(f, iterate(f.itr))
iterate(f::Filter, s) = filternext(f, iterate(f.itr, s))
zipnext(::Nothing, b) = nothing
zipnext(a::Tuple, ::Nothing) = nothing
zipnext(a::Tuple, b::Tuple) = (Pair(getfield(a, 1), getfield(b, 1)), Pair(getfield(a, 2), getfield(b, 2)))
iterate(z::Zip) = zipnext(iterate(z.a), iterate(z.b))
iterate(z::Zip, s::Pair) = zipnext(iterate(z.a, s.first), iterate(z.b, s.second))
enumnext(i::Int, ::Nothing) = nothing
enumnext(i::Int, it::Tuple) = (Pair(i, getfield(it, 1)), Pair(i + 1, getfield(it, 2)))
iterate(e::Enumerate) = enumnext(1, iterate(e.itr))
iterate(e::Enumerate, s::Pair) = enumnext(s.first, iterate(e.itr, s.second))

# The list of the elements of an iterator, of the type of its first one
function collect(itr)
    it = iterate(itr)
    it === nothing && error(:empty)
    x = getfield(it, 1)
    T = typeof(x)
    acc = @new(Cons{T}(x, Nil{T}()))
    it = iterate(itr, getfield(it, 2))
    while it !== nothing
        acc = @new(Cons{T}(getfield(it, 1), acc))
        it = iterate(itr, getfield(it, 2))
    end
    return reverse(acc)
end
function mapfoldl(f, op, acc, itr)
    for x in itr
        acc = op(acc, f(x))
    end
    return acc
end
function mapreduce(f, op, itr)
    it = iterate(itr)
    it === nothing && error(:empty)
    acc = f(getfield(it, 1))
    it = iterate(itr, getfield(it, 2))
    while it !== nothing
        acc = op(acc, f(getfield(it, 1)))
        it = iterate(itr, getfield(it, 2))
    end
    return acc
end
sum(f, itr) = mapreduce(f, +, itr)
sum(itr) = mapreduce(identity, +, itr)
maximum(itr) = mapreduce(identity, max, itr)
extrema(itr) = mapreduce(x -> (x, x), (a, b) -> (min(getfield(a, 1), getfield(b, 1)), max(getfield(a, 2), getfield(b, 2))), itr)
count(pred, itr) = mapfoldl(x -> pred(x) ? 1 : 0, +, 0, itr)
function any(pred, itr)
    for x in itr
        pred(x) && return true
    end
    return false
end
function findall(pred, itr)
    acc = Nil{Int}()
    for p in Enumerate(itr)
        if pred(p.second)
            acc = @new(Cons{Int}(p.first, acc))
        end
    end
    return reverse(acc)
end

## A search tree

abstract type Tree{K,V} <: Obj end
struct Leaf{K,V} <: Tree{K,V} end
struct Node{K,V} <: Tree{K,V}
    key::K
    value::V
    left::Tree{K,V}
    right::Tree{K,V}
end
insert(t::Leaf{K,V}, k::K, v::V) where {K,V} = @new(Node{K,V}(k, v, t, t))
function insert(t::Node{K,V}, k::K, v::V) where {K,V}
    if isless(k, t.key)
        return @new(Node{K,V}(t.key, t.value, insert(t.left, k, v), t.right))
    elseif isless(t.key, k)
        return @new(Node{K,V}(t.key, t.value, t.left, insert(t.right, k, v)))
    else
        return @new(Node{K,V}(k, v, t.left, t.right))
    end
end
get(t::Leaf, k, default) = default
function get(t::Node, k, default)
    isless(k, t.key) && return get(t.left, k, default)
    isless(t.key, k) && return get(t.right, k, default)
    return t.value
end
haskey(t::Tree, k) = get(t, k, haskey) !== haskey
# In order, with the stack of nodes still to visit as the state
descend(t::Leaf, stack::List) = stack
descend(t::Node{K,V}, stack::List{Node{K,V}}) where {K,V} = descend(t.left, @new(Cons{Node{K,V}}(t, stack)))
iterate(t::Tree{K,V}) where {K,V} = treenext(descend(t, Nil{Node{K,V}}()))
iterate(t::Tree, stack::List) = treenext(stack)
treenext(stack::Nil) = nothing
treenext(stack::Cons) = (Pair(stack.head.key, stack.head.value), descend(stack.head.right, stack.tail))
function tree(pairs::List{Pair{K,V}}) where {K,V}
    t = Leaf{K,V}()
    for p in pairs
        t = insert(t, p.first, p.second)
    end
    return t
end

## Sorting

abstract type Ordering <: Obj end
struct Forward <: Ordering end
struct Reverse{O<:Ordering} <: Ordering
    fwd::O
end
struct By{F,O<:Ordering} <: Ordering
    by::F
    order::O
end
struct Lt{F} <: Ordering
    lt::F
end
lt(::Forward, a, b) = isless(a, b)
lt(o::Reverse, a, b) = lt(o.fwd, b, a)
lt(o::By, a, b) = lt(o.order, o.by(a), o.by(b))
lt(o::Lt, a, b) = o.lt(a, b)
# The ordering of `sort(l, lt, by, rev)`, of a type that depends on the values
ord(lt, by, rev::Bool) = rev ? Reverse(ord(lt, by)) : ord(lt, by)
ord(::typeof(isless), ::typeof(identity)) = Forward()
ord(::typeof(isless), by) = By(by, Forward())
ord(lt, ::typeof(identity)) = Lt(lt)
ord(lt, by) = By(by, Lt(lt))

function merge(a::List{T}, b::List{T}, o::Ordering) where {T}
    acc = Nil{T}()
    while !isempty(a) & !isempty(b)
        if lt(o, b.head, a.head)
            acc = @new(Cons{T}(b.head, acc))
            b = b.tail
        else
            acc = @new(Cons{T}(a.head, acc))
            a = a.tail
        end
    end
    for x in a
        acc = @new(Cons{T}(x, acc))
    end
    for x in b
        acc = @new(Cons{T}(x, acc))
    end
    return reverse(acc)
end
function insertion_sort(l::List{T}, o::Ordering) where {T}
    sorted = Nil{T}()
    for x in l
        front = Nil{T}()
        rest = sorted
        while !isempty(rest) && !lt(o, x, rest.head)
            front = @new(Cons{T}(rest.head, front))
            rest = rest.tail
        end
        rest = @new(Cons{T}(x, rest))
        for y in front
            rest = @new(Cons{T}(y, rest))
        end
        sorted = rest
    end
    return sorted
end
function sort(l::List{T}, o::Ordering) where {T}
    n = length(l)
    n <= 8 && return insertion_sort(l, o)
    half = div(n, 2)
    # Inference widens the recursive calls
    return merge(typeassert(sort(take(l, half), o), List{T}), typeassert(sort(drop(l, half), o), List{T}), o)
end
sort(l::List) = sort(l, Forward())
sort(l::List, lt, by, rev::Bool) = sort(l, ord(lt, by, rev))
function issorted(l::List, o::Ordering)
    it = iterate(l)
    it === nothing && return true
    prev = getfield(it, 1)
    it = iterate(l, getfield(it, 2))
    while it !== nothing
        lt(o, getfield(it, 1), prev) && return false
        prev = getfield(it, 1)
        it = iterate(l, getfield(it, 2))
    end
    return true
end
sortperm(l::List, o::Ordering) = collect(Map(first, sort(collect(Enumerate(l)), By(second, o))))

## Printing, into a list of tokens

mutable struct Buf <: Obj
    out::List{Any}
    n::Int
end
Buf() = Buf(Nil{Any}(), 0)
function write(io::Buf, Base.@nospecialize(token))
    io.out = @new(Cons{Any}(token, io.out))
    io.n += 1
    return nothing
end
print(io::Buf) = nothing
print(io::Buf, x) = show(io, x)
print(io::Buf, s::Union{String,Symbol,Char}) = write(io, s)
print(io::Buf, a, b, rest...) = (print(io, a); print(io, b, rest...))
println(io::Buf, xs...) = (print(io, xs...); write(io, '\n'))

function showdigits(io::Buf, x::UInt64, base::UInt64)
    digits = Nil{UInt8}()
    while true
        d = trunc_int(UInt8, rem(x, base))
        digits = @new(Cons{UInt8}(d < 0x0a ? 0x30 + d : 0x57 + d, digits))
        x = div(x, base)
        iszero(x) && break
    end
    for d in digits
        write(io, d)
    end
end
function show(io::Buf, x::Int)
    x < 0 && write(io, '-')
    showdigits(io, bitcast(UInt64, abs(x)), 0x000000000000000a)
end
show(io::Buf, x::UInt64) = (write(io, "0x"); showdigits(io, x, 0x0000000000000010))
show(io::Buf, x::UInt8) = (write(io, "0x"); showdigits(io, convert(UInt64, x), 0x0000000000000010))
function show(io::Buf, x::Float64)
    x == x || return write(io, "NaN")
    if x < 0.0
        write(io, '-')
        x = -x
    end
    x < 9.0e18 || return write(io, "Inf")
    i = trunc(Int, x)
    show(io, i)
    write(io, '.')
    f = x - convert(Float64, i)
    for _ in 1:6
        f = f * 10.0
        d = trunc(Int, f)
        write(io, 0x30 + convert(UInt8, d))
        f = f - convert(Float64, d)
    end
end
show(io::Buf, x::Bool) = write(io, x ? "true" : "false")
show(io::Buf, ::Nothing) = write(io, "nothing")
show(io::Buf, s::Symbol) = (write(io, ':'); write(io, s))
show(io::Buf, s::String) = (write(io, '"'); write(io, s); write(io, '"'))
show(io::Buf, c::Char) = (write(io, '\''); write(io, c); write(io, '\''))
show(io::Buf, T::Type) = write(io, T)
show(io::Buf, x::Rat) = (show(io, x.num); write(io, "//"); show(io, x.den))
show(io::Buf, z::Cplx) = (show(io, z.re); write(io, " + "); show(io, z.im); write(io, "im"))
show(io::Buf, x::Dual) = (write(io, "Dual("); show(io, x.val); write(io, ", "); show(io, x.der); write(io, ')'))
show(io::Buf, r::Range) = (show(io, r.start); write(io, ':'); show(io, r.stop))
show(io::Buf, p::Pair) = (show(io, p.first); write(io, " => "); show(io, p.second))
show_items(io::Buf, t::Tuple{}) = nothing
show_items(io::Buf, t::Tuple{Any}) = show(io, first(t))
show_items(io::Buf, t::Tuple) = (show(io, first(t)); write(io, ", "); show_items(io, tail(t)))
show(io::Buf, t::Tuple) = (write(io, '('); show_items(io, t); write(io, ')'))
function show_delim(io::Buf, itr, open::Char, close::Char)
    write(io, open)
    isfirst = true
    for x in itr
        isfirst || write(io, ", ")
        isfirst = false
        show(io, x)
    end
    write(io, close)
end
show(io::Buf, l::List) = show_delim(io, l, '[', ']')
show(io::Buf, v::Vec) = (write(io, "Vec"); show_delim(io, v.items, '[', ']'))
show(io::Buf, t::Tree) = (write(io, "Tree"); show_delim(io, t, '(', ')'))
# Any other value, by its fields
function show(io::Buf, Base.@nospecialize(x))
    write(io, typeof(x))
    write(io, '(')
    i = 1
    while i <= nfields(x)
        i > 1 && write(io, ", ")
        show(io, getfield(x, i))
        i += 1
    end
    write(io, ')')
end

## A small expression tree and its printing

const Literal = Union{Int,Float64,Bool,Rat{Int}}
struct Lit <: Obj
    value::Literal
end
struct Var <: Obj
    name::Symbol
end
struct Call <: Obj
    head::Symbol
    args::List{Any}
end
struct Quote <: Obj
    value::Any
end

function precedence(op::Symbol)
    op === :(=) && return 1
    (op === :|| || op === :&&) && return 4
    (op === :< || op === :> || op === :(==) || op === :<= || op === :>=) && return 7
    (op === :+ || op === :-) && return 11
    (op === :* || op === :/) && return 12
    op === :^ && return 15
    return 0
end
function show_indent(io::Buf, indent::Int)
    for _ in 1:indent
        write(io, ' ')
    end
end
function show_args(io::Buf, args::List{Any}, indent::Int, open::Char, close::Char)
    write(io, open)
    isfirst = true
    for a in args
        isfirst || write(io, ", ")
        isfirst = false
        show_unquoted(io, a, indent, 0)
    end
    write(io, close)
end
function show_block(io::Buf, args::List{Any}, indent::Int)
    for a in args
        show_indent(io, indent + 4)
        show_unquoted(io, a, indent + 4, 0)
        write(io, '\n')
    end
    show_indent(io, indent)
end
function show_call(io::Buf, ex::Call, indent::Int, prec::Int)
    head = ex.head
    args = ex.args
    nargs = length(args)
    if head === :call && nargs >= 1
        f = args[1]
        op = f isa Var ? precedence(f.name) : 0
        if (op > 0) & (nargs == 3)
            op < prec && write(io, '(')
            show_unquoted(io, args[2], indent, op)
            write(io, ' ')
            show_unquoted(io, f, indent, 0)
            write(io, ' ')
            show_unquoted(io, args[3], indent, op + 1)
            op < prec && write(io, ')')
        elseif (op > 0) & (nargs == 2)
            show_unquoted(io, f, indent, 0)
            show_unquoted(io, args[2], indent, 14)
        else
            show_unquoted(io, f, indent, 16)
            show_args(io, drop(args, 1), indent, '(', ')')
        end
    elseif (head === :(=)) & (nargs == 2)
        show_unquoted(io, args[1], indent, 1)
        write(io, " = ")
        show_unquoted(io, args[2], indent, 1)
    elseif head === :tuple
        show_args(io, args, indent, '(', ')')
    elseif head === :ref && nargs >= 1
        show_unquoted(io, args[1], indent, 16)
        show_args(io, drop(args, 1), indent, '[', ']')
    elseif head === :block
        write(io, "begin\n")
        show_block(io, args, indent)
        write(io, "end")
    elseif (head === :if) & (nargs >= 2)
        write(io, "if ")
        show_unquoted(io, args[1], indent, 0)
        write(io, '\n')
        show_block(io, @new(Cons{Any}(args[2], Nil{Any}())), indent)
        if nargs == 3
            write(io, "else\n")
            show_block(io, @new(Cons{Any}(args[3], Nil{Any}())), indent)
        end
        write(io, "end")
    elseif (head === :return) & (nargs == 1)
        write(io, "return ")
        show_unquoted(io, args[1], indent, 0)
    else
        write(io, "Expr")
        show_args(io, @new(Cons{Any}(Quote(head), args)), indent, '(', ')')
    end
end
function show_quote(io::Buf, Base.@nospecialize(value), indent::Int)
    if value isa Symbol
        write(io, ':')
        write(io, value)
    elseif value isa Var
        write(io, ':')
        write(io, value.name)
    elseif value isa Call
        write(io, ":(")
        show_call(io, value, indent, 0)
        write(io, ')')
    else
        write(io, ":(")
        show_unquoted(io, value, indent, 0)
        write(io, ')')
    end
end
function show_unquoted(io::Buf, Base.@nospecialize(ex), indent::Int, prec::Int)
    if ex isa Call
        show_call(io, ex, indent, prec)
    elseif ex isa Var
        write(io, ex.name)
    elseif ex isa Lit
        show(io, ex.value)
    elseif ex isa Quote
        show_quote(io, ex.value, indent)
    elseif ex isa Symbol
        write(io, ex)
    else
        show(io, ex)
    end
end
show(io::Buf, ex::Union{Lit,Var,Call}) = show_quote(io, ex, 0)
show(io::Buf, ex::Quote) = show_quote(io, ex.value, 0)

## A lazy broadcast

struct Broadcasted{F,Args<:Tuple} <: Obj
    f::F
    args::Args
end
broadcasted(f, args...) = Broadcasted(f, args)
# The common length of the arguments, where 0 is that of a scalar
bclength(x::Number) = 0
bclength(v::Vec) = length(v)
bclength(bc::Broadcasted) = foldl(bccombine, 0, map(bclength, bc.args))
bccombine(a::Int, b::Int) = a == 0 ? b : ((b == 0) | (a == b) ? a : error(:dimension_mismatch))
bcget(x::Number, i::Int) = x
bcget(v::Vec, i::Int) = v[i]
bcget(bc::Broadcasted, i::Int) = bc.f(map(Fix2(bcget, i), bc.args)...)
materialize(bc::Broadcasted) = vec(collect(Map(Fix1(bcget, bc), 1:bclength(bc))))

## A random number generator (xoshiro256++)

mutable struct Xoshiro <: Obj
    s0::UInt64
    s1::UInt64
    s2::UInt64
    s3::UInt64
end
rotl(x::UInt64, k::Int) = (x << k) | (x >> (64 - k))
function next!(rng::Xoshiro)
    s0 = rng.s0
    s1 = rng.s1
    s2 = rng.s2
    s3 = rng.s3
    res = rotl(s0 + s3, 23) + s0
    t = s1 << 17
    s2 = xor(s2, s0)
    s3 = xor(s3, s1)
    s1 = xor(s1, s2)
    s0 = xor(s0, s3)
    s2 = xor(s2, t)
    rng.s0 = s0
    rng.s1 = s1
    rng.s2 = s2
    rng.s3 = rotl(s3, 45)
    return res
end
rand(rng::Xoshiro, ::Type{UInt64}) = next!(rng)
rand(rng::Xoshiro, ::Type{UInt8}) = trunc_int(UInt8, next!(rng) >> 56)
rand(rng::Xoshiro, ::Type{Int}) = bitcast(Int, next!(rng))
rand(rng::Xoshiro, ::Type{Bool}) = (next!(rng) >> 63) == 0x0000000000000001
rand(rng::Xoshiro, ::Type{Float64}) = convert(Float64, next!(rng) >> 11) * 0x1.0p-53
rand(rng::Xoshiro, ::Type{Cplx{T}}) where {T<:Real} = Cplx(rand(rng, T), rand(rng, T))
rand(rng::Xoshiro, ::Type{Rat{Int}}) = rand(rng, -99:99) // rand(rng, 1:99)
rand(rng::Xoshiro, ::Type{Pair{A,B}}) where {A,B} = Pair(rand(rng, A), rand(rng, B))
rand(rng::Xoshiro) = rand(rng, Float64)
# Without bias, by rejection
function rand(rng::Xoshiro, r::Range)
    n = convert(UInt64, length(r))
    iszero(n) && error(:empty)
    limit = 0xffffffffffffffff - rem(0xffffffffffffffff, n)
    x = next!(rng)
    while x >= limit
        x = next!(rng)
    end
    return r.start + convert(Int, rem(x, n))
end
rand(rng::Xoshiro, l::List) = l[rand(rng, 1:length(l))]
function randn(rng::Xoshiro)
    while true
        u = 2.0 * rand(rng) - 1.0
        v = 2.0 * rand(rng) - 1.0
        s = u * u + v * v
        ((s < 1.0) & (s > 0.0)) && return u * sqrt(-2.0 * log(s) / s)
    end
end
function rand(rng::Xoshiro, ::Type{T}, n::Int) where {T}
    acc = Nil{T}()
    for _ in 1:n
        acc = @new(Cons{T}(rand(rng, T), acc))
    end
    return acc
end
shuffle(rng::Xoshiro, l::List) = collect(Map(second, sort(collect(Map(x -> Pair(rand(rng, UInt64), x), l)), By(first, Forward()))))

## What the benchmarks infer

# Promotion between number types, the elementary functions and their derivatives
function numeric_kernel(n::Int)
    x = convert(Float64, n) / 8
    r = n // 12 + 3 // 4
    z = Cplx(x, 1) * Cplx(r, r) + exp(Cplx(0.0, x))
    d = derivative(t -> sin(t * t) * exp(-t) + sqrt(t + 2) / (1 + t * t), x)
    p = evalpoly(Cplx(r, 1 // 2), (1, 2.5, r, Cplx(1, 2)))
    q = evalpoly(Dual(x, 1.0), (r, 2, 0.5, n))
    s = foldl(+, 0, (n, x, r, z, d, 0x03))
    t = map(abs, (n, x, r, z, q))
    return (s, p, q, t, ntuple(i -> i * r, Val{4}()), log(x + 2.0) < cos(r))
end
# Printing of an expression tree whose leaves are of any type
show_kernel(io::Buf, q::Quote) = println(io, "ex = ", q, ' ', precedence(:+))
# Printing of a tree of lists, of tuples and of numbers
show_collections_kernel(io::Buf, t::Tree{Tuple{Int,Rat{Int}},List{Cplx{Float64}}}, v::Vec{Pair{Symbol,Dual{Float64}}}) =
    println(io, t, " and ", v, (1, 2.0, 0x03, 1 // 2, nothing, "s", 'c', :sym, 1:3), Pair(Val{1}(), Float64))
# Nested lazy broadcasts with scalars of other types, and a search in the result
function broadcast_kernel(xs::Vec{Float64}, ys::Vec{Rat{Int}}, x::Float64)
    a = materialize(broadcasted(abs, broadcasted(-, broadcasted(*, xs, 2), x)))
    b = materialize(broadcasted(muladd, ys, 1 // 2, broadcasted(sin, xs)))
    c = materialize(broadcasted(cplx, a, b))
    return (findall(Fix2(>, x), a), sum(abs2, c), extrema(b), count(z -> z.re < z.im, c))
end
# Sampling of several types, rejection, the normal distribution and a shuffle
function rand_kernel(rng::Xoshiro)
    a = rand(rng, Float64, 8)
    b = rand(rng, Pair{Int,Cplx{Rat{Int}}}, 4)
    c = shuffle(rng, rand(rng, UInt8, 16))
    d = (rand(rng), randn(rng), rand(rng, Bool), rand(rng, 1:6), rand(rng, a), rand(rng, Rat{Int}))
    return (sum(a), b, c, d)
end
# Sorting by orderings built from functions, of a type that depends on a value
function sort_kernel(xs::List{Float64}, ps::List{Pair{Int,Rat{Int}}}, rev::Bool)
    a = sort(xs, isless, abs, rev)
    b = sort(ps, (p, q) -> p.second < q.second, identity, rev)
    c = sortperm(xs, Reverse(Forward()))
    d = sort(collect(Zip(xs, ps)), By(p -> (p.second.first, -p.first), Forward()))
    return (a, b, c, d, issorted(a, Forward()), maximum(xs), any(p -> haskey(tree(ps), p.first), ps))
end

## The stress tests' callees

strangesum(::Vec{Float64}) = error(:not_called)
strangesum(x::Obj) = sum(x)
const constprop_sink = Xoshiro(0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000)
Base.@constprop :aggressive function constprop_target(x::Float64)
    constprop_sink.s0 = bitcast(UInt64, x)
    return x > 0.0 ? x : -x
end

end
