# Code that is not fully inferred: runtime dispatch, union splitting, boxed closures,
# keyword calls and splatting. Receivers are shuffled so call targets are unpredictable.
module DispatchBench
using ..Driver

abstract type Shape end
struct Circle <: Shape; r::Float64; end
struct Square <: Shape; a::Float64; end
struct Rect <: Shape; a::Float64; b::Float64; end
struct Tri <: Shape; b::Float64; h::Float64; end
area(s::Circle) = 3.14159 * s.r^2
area(s::Square) = s.a^2
area(s::Rect) = s.a * s.b
area(s::Tri) = 0.5 * s.b * s.h

function shapes_input()
    r = SplitMix(1)
    x = rand_f64(r, 2^16)
    kinds = rand_int(r, 2^16, 1, 4)
    return Shape[k == 1 ? Circle(v) : k == 2 ? Square(v) : k == 3 ? Rect(v, 2v) : Tri(v, 3v) for (k, v) in zip(kinds, x)]
end

# Abstract element type: one dynamic dispatch per element through the method cache
bench("abstract_eltype_call"; ops = 2^16, setup = shapes_input) do shapes
    s = 0.0
    for sh in shapes
        s += area(sh)
    end
    s
end

# A small Union element type stored inline: union-split, no dispatch
function union_input()
    r = SplitMix(2)
    k = rand_int(r, 2^17, 1, 3)
    return Union{Int,Float64,Nothing}[i == 1 ? 1 : i == 2 ? 0.5 : nothing for i in k]
end
bench("union_split"; ops = 2^17, setup = union_input) do xs
    s = 0.0
    for x in xs
        if x !== nothing
            s += x
        end
    end
    s
end

# Fully dynamic arithmetic with boxing: Vector{Any} and an untyped accumulator
bench("any_arith"; ops = 2^15, setup = () -> Any[isodd(i) ? i : Float64(i) for i in rand_int(SplitMix(3), 2^15, 1, 1000)]) do xs
    s = 0
    for x in xs
        s = s + x * 2
    end
    s
end

# A captured variable that is reassigned becomes a Core.Box; lowering decides this
function boxed_closure(xs)
    acc = 0.0
    f = x -> (acc += x; acc)
    for x in xs
        f(x)
    end
    return acc
end
bench("boxed_closure"; ops = 2^16, setup = () -> rand_f64(SplitMix(4), 2^16)) do xs
    boxed_closure(xs)
end

# Keyword calls with runtime keyword sets: kwcall, NamedTuple merging, the kwsorter
kwf(x; scale = 1.0, shift = 0.0, flip::Bool = false) = (flip ? -x : x) * scale + shift
function kwcall_input()
    kws = Any[(; scale = 2.0), (; shift = 1.0), (; scale = 0.5, flip = true), (;), (; flip = false, shift = -1.0)]
    return rand_f64(SplitMix(5), 2^14), kws
end
bench("kwcall_dynamic"; ops = 2^14, setup = kwcall_input) do (xs, kws)
    s = 0.0
    for (i, x) in enumerate(xs)
        s += kwf(x; kws[1 + i % length(kws)]...)
    end
    s
end

# Splatting runtime-length collections goes through Core._apply_iterate
function splat_input()
    r = SplitMix(6)
    return [rand_f64(r, 1 + i % 7) for i in 1:2^13]
end
bench("splat_dynamic"; ops = 2^13, setup = splat_input) do vs
    s = 0.0
    for v in vs
        s += max(v...) + +(0.0, v...)
    end
    s
end

# invokelatest: the world-age escape hatch used by REPL, callbacks and plugins
bench("invokelatest"; ops = 2^14, setup = () -> rand_f64(SplitMix(7), 2^14)) do xs
    s = 0.0
    for x in xs
        s += Base.invokelatest(muladd, x, 2.0, 1.0)
    end
    s
end

end
