# Code that is not fully inferred: runtime dispatch, union splitting, boxed closures,
# keyword calls and splatting. Receivers are shuffled so call targets are unpredictable.
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
        s += area(sh)
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

# Fully dynamic arithmetic with boxing: Vector{Any} and an untyped accumulator
any_arith_setup() = Any[isodd(i) ? i : Float64(i) for i in rand_int(SplitMix(3), 2^15, 1, 1000)]
function any_arith(xs)
    s = 0
    for x in xs
        s = s + x * 2
    end
    return s
end
bench(any_arith, any_arith_setup; ops = 2^15)

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

# Keyword calls with runtime keyword sets: kwcall, NamedTuple merging, the kwsorter
function kwcall_dynamic_setup()
    r = SplitMix(5)
    kws = Any[(; scale = 2.0), (; shift = 1.0), (; scale = 0.5, flip = true), (;), (; flip = false, shift = -1.0)]
    return rand_f64(r, 2^14), kws, rand_int(r, 2^14, 1, 5)
end
kwf(x; scale = 1.0, shift = 0.0, flip::Bool = false) = (flip ? -x : x) * scale + shift
function kwcall_dynamic(xs, kws, which)
    s = 0.0
    for i in eachindex(xs, which)
        s += kwf(xs[i]; kws[which[i]]...)
    end
    return s
end
bench(kwcall_dynamic, kwcall_dynamic_setup; ops = 2^14)

# Splatting runtime-length collections goes through Core._apply_iterate
function splat_dynamic_setup()
    r = SplitMix(6)
    return [rand_f64(r, n) for n in rand_int(r, 2^13, 1, 8)]
end
function splat_dynamic(vs)
    s = 0.0
    for v in vs
        s += max(v...) + +(0.0, v...)
    end
    return s
end
bench(splat_dynamic, splat_dynamic_setup; ops = 2^13)

# invokelatest: the world-age escape hatch used by REPL, callbacks and plugins
invokelatest_calls_setup() = rand_f64(SplitMix(7), 2^14)
function invokelatest_calls(xs)
    s = 0.0
    for x in xs
        s += Base.invokelatest(muladd, x, 2.0, 1.0)
    end
    return s
end
bench(invokelatest_calls, invokelatest_calls_setup; ops = 2^14)

end
