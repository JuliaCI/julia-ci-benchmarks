# Code that is not fully inferred: runtime dispatch, union splitting, boxed closures,
# keyword calls and splatting. Receivers are shuffled so call targets are unpredictable.
# Dynamic calls go through a call cache keyed on the caller's address, so their
# instruction counts move a few percent with code placement: those benchmarks use 6%.
module DispatchBench
using ..Driver

abstract type Shape end
struct Circle <: Shape; r::Float64; end
struct Square <: Shape; a::Float64; end
struct Rect <: Shape; a::Float64; b::Float64; end
struct Tri <: Shape; b::Float64; h::Float64; end

# Abstract element type: one dynamic dispatch per element through the method cache
function abstract_eltype_call_setup()
    r = SplitMix(1)
    x = rand_f64(r, 2^16)
    kinds = rand_int(r, 2^16, 1, 4)
    return Shape[k == 1 ? Circle(v) : k == 2 ? Square(v) : k == 3 ? Rect(v, 2v) : Tri(v, 3v) for (k, v) in zip(kinds, x)]
end
area(s::Circle) = 3.14159 * s.r^2
area(s::Square) = s.a^2
area(s::Rect) = s.a * s.b
area(s::Tri) = 0.5 * s.b * s.h
function abstract_eltype_call(shapes)
    s = 0.0
    for sh in shapes
        s += area(sh)::Float64
    end
    return s
end
bench(abstract_eltype_call, abstract_eltype_call_setup; ops = 2^16)

# A small Union element type stored inline: union-split, no dispatch
function union_split_loop_setup()
    r = SplitMix(2)
    k = rand_int(r, 2^17, 1, 3)
    return Union{Int,Float64,Nothing}[i == 1 ? 1 : i == 2 ? 0.5 : nothing for i in k]
end
function union_split_loop(xs)
    s = 0.0
    for x in xs
        if x !== nothing
            s += x
        end
    end
    return s
end
bench(union_split_loop, union_split_loop_setup; ops = 2^17)

# An isbits Union returned from a call that is not inlined: the union return ABI (payload
# buffer plus type tag) and union splitting at the caller
struct SmallResult
    x::Float64
end
struct WideResult
    x::Float64
    y::Float64
end
union_return_abi_setup() = rand_f64(SplitMix(8), 2^16, -1, 1)
@noinline choose_result(x) = x < 0 ? SmallResult(-x) : WideResult(x, x * x)
function union_return_abi(xs)
    s = 0.0
    @inbounds for x in xs
        r = choose_result(x)
        s += r isa SmallResult ? r.x : r.x + r.y
    end
    return s
end
bench(union_return_abi, union_return_abi_setup; ops = 2^16)

# Fully dynamic arithmetic with boxing: Vector{Any} and an untyped accumulator. The
# elements are reshuffled every sample.
any_arith_setup() = (Any[isodd(i) ? i : Float64(i) for i in rand_int(SplitMix(3), 2^15, 1, 1000)], SplitMix(203))
function any_arith(xs, _)
    s = 0
    for x in xs
        s = s + x * 2
    end
    return s
end
bench(any_arith, any_arith_setup; ops = 2^15, tolerance = 0.06, reset = (xs, r) -> shuffle_det!(r, xs))

# A captured variable that is reassigned becomes a Core.Box; lowering decides this
boxed_closure_setup() = rand_f64(SplitMix(4), 2^16)
function boxed_closure(xs)
    acc = 0.0
    f = x -> (acc += x; acc)
    for x in xs
        f(x)
    end
    return acc
end
bench(boxed_closure, boxed_closure_setup; ops = 2^16)

# Keyword calls with runtime keyword sets: kwcall, NamedTuple merging, the kwsorter. The
# (value, keyword set) pairs are reshuffled every sample.
function kwcall_dynamic_setup()
    r = SplitMix(5)
    kws = Any[(; scale = 2.0), (; shift = 1.0), (; scale = 0.5, flip = true), (;), (; flip = false, shift = -1.0)]
    xs, which = carve(Float64 => 2^14, Int => 2^14)
    xs .= rand_f64(r, 2^14)
    which .= rand_int(r, 2^14, 1, 5)
    return xs, kws, which, SplitMix(205)
end
kwf(x; scale = 1.0, shift = 0.0, flip::Bool = false) = (flip ? -x : x) * scale + shift
function kwcall_dynamic(xs, kws, which, _)
    s = 0.0
    for i in eachindex(xs, which)
        s += kwf(xs[i]; kws[which[i]]...)
    end
    return s
end
bench(kwcall_dynamic, kwcall_dynamic_setup; ops = 2^14, tolerance = 0.06,
      reset = (xs, kws, which, r) -> shuffle_det!(r, xs, which))

# Splatting runtime-length collections goes through Core._apply_iterate. The vectors are
# reshuffled every sample, so the lengths do not repeat in a learnable order.
function splat_dynamic_setup()
    r = SplitMix(6)
    return [rand_f64(r, n) for n in rand_int(r, 2^13, 1, 8)], SplitMix(206)
end
function splat_dynamic(vs, _)
    s = 0.0
    for v in vs
        s += max(v...) + +(0.0, v...)
    end
    return s
end
bench(splat_dynamic, splat_dynamic_setup; ops = 2^13, tolerance = 0.06, reset = (vs, r) -> shuffle_det!(r, vs))

# invokelatest: the world-age escape hatch used by REPL, callbacks and plugins
invokelatest_calls_setup() = rand_f64(SplitMix(7), 2^14)
function invokelatest_calls(xs)
    s = 0.0
    for x in xs
        s += Base.invokelatest(muladd, x, 2.0, 1.0)
    end
    return s
end
bench(invokelatest_calls, invokelatest_calls_setup; ops = 2^14, tolerance = 0.06)


# An accumulator that starts as an Int and becomes a Float64: a Union{Int,Float64}
# carried around the loop (#52875)
union_accumulator_setup() = rand_f64(SplitMix(34), 2^16)
function union_accumulator(x)
    s = 0
    for t in x
        s += t
    end
    return s
end
bench(union_accumulator, union_accumulator_setup; ops = 2^16)

# A Vector of a union of six singleton types through sum(f, v) and a generator (#58370).
# Past the union-splitting limit in principle; in practice the weights become a table
# lookup and the reduction vectorizes, which is what this guards
struct S1 end
struct S2 end
struct S3 end
struct S4 end
struct S5 end
struct S6 end
weight(::S1) = 1
weight(::S2) = 2
weight(::S3) = 3
weight(::S4) = 4
weight(::S5) = 5
weight(::S6) = 6
function big_union_setup()
    inst = (S1(), S2(), S3(), S4(), S5(), S6())
    return Union{S1,S2,S3,S4,S5,S6}[inst[k] for k in rand_int(SplitMix(35), 2^18, 1, 6)]
end
big_union(v) = sum(weight, v) + sum(weight(x) for x in v)
bench(big_union, big_union_setup; ops = 2 * 2^18)

# Iterating a tuple of different functions (dynamic, allocating), and recursing over one
# (inlined; the shape of #58168)
const BINOPS = (+, -, *, /, hypot, atan)
@inline fmap_rec(x, f, fs...) = (f(x...), fmap_rec(x, fs...)...)
@inline fmap_rec(x) = ()
function function_tuple_setup()
    r = SplitMix(36)
    return rand_f64(r, 2^12, 0.5, 2), rand_f64(r, 2^12, 0.5, 2)
end
function function_tuple(xs, ys)
    s = 0.0
    for i in eachindex(xs)
        x, y = xs[i], ys[i]
        for op in BINOPS
            s += op(x, y)
        end
        s += sum(fmap_rec((x, y), hypot, atan, min, max, copysign))
    end
    return s
end
bench(function_tuple, function_tuple_setup; ops = 2^12, tolerance = 0.06)

# A parametric constructor called on values of unknown type: dispatch on Type{Box},
# apply_type and allocation at runtime (#60701)
struct Box{T}
    a::T
end
function dynamic_ctor_setup()
    r = SplitMix(37)
    vals = Any[1, 2.0, "s", true, 'c', 0x01, Int32(3), 1.5f0]
    xs = Any[vals[k] for k in rand_int(r, 2^14, 1, 8)]
    return xs, similar(xs), SplitMix(237)
end
# The values are reshuffled every sample, so the constructed type does not repeat in a
# learnable order
function dynamic_ctor(xs, out, _)
    for i in eachindex(xs, out)
        out[i] = Box(xs[i])
    end
    return out
end
bench(dynamic_ctor, dynamic_ctor_setup; ops = 2^14, tolerance = 0.06, reset = (xs, out, r) -> shuffle_det!(r, xs))

end
