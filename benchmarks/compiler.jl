# Compiler throughput on fixed inputs: parsing, macro expansion, lowering, inference,
# LLVM and invalidation. This complements the TTFX suite, which measures whole packages:
# here the input never changes, so a difference is the compiler's own. Every sample works
# on code defined into a fresh module (in the untimed reset), so no cache hides the work.
module CompilerBench
using ..Driver

# Shared by the benchmarks below: paste this part first to run one of them in the REPL.
const SRC = read(joinpath(Driver.DATA_DIR, "compiler_workload.jl"), String)

fresh_module() = Module(:CompilerWorkload, true)

function load(src)
    m = fresh_module()
    Base.include_string(m, src, "workload.jl")
    return m
end
latest(m, name) = Base.invokelatest(getfield, m, name)
# One long function of straight-line arithmetic and branches (lower_large_function,
# llvm_large_function): little to infer, a lot for lowering and LLVM
function large_function_src()
    io = IOBuffer()
    println(io, "function big(x::Float64, y::Float64, n::Int)")
    r = SplitMix(1)
    for i in 1:300
        a, b = rand_int(r, 2, 0, i - 1)
        va = a == 0 ? "x" : "v$a"
        vb = b == 0 ? "y" : "v$b"
        k = i % 5
        if k == 0
            println(io, "    v$i = $va * $vb + $(i).5")
        elseif k == 1
            println(io, "    v$i = n > $i ? $va - $vb : $vb / ($va + 1.5)")
        elseif k == 2
            println(io, "    v$i = muladd($va, $(i % 7 + 1).0, $vb)")
        elseif k == 3
            println(io, "    v$i = ifelse($va < $vb, sqrt(abs($va)), $vb * $vb)")
        else
            println(io, "    v$i = $va + (n % $(i % 11 + 2)) * $vb")
        end
    end
    println(io, "    return v300 + v299")
    println(io, "end")
    return String(take!(io))
end
const BIG_SRC = large_function_src()
# End of the shared part.

bench("parse"; ops = sizeof(SRC), setup = () -> SRC) do src
    length(Meta.parseall(src).args)
end

bench("lower"; setup = () -> (Meta.parseall(SRC), Ref{Module}()), reset = st -> (st[2][] = fresh_module())) do (ex, mref)
    n = 0
    for e in ex.args
        e isa LineNumberNode && continue
        n += length(string(typeof(Meta.lower(mref[], e))))
    end
    n
end

# Lowering passes the workload touches lightly: closure conversion, keyword-method
# generation, destructuring, do-blocks, comprehensions and try/finally
const LOWER_SRC = """
function kwheavy(a, b = 2, c...; scale::Float64 = 1.0, offset = 0, flags::Tuple = (), kws...)
    (x, y), z = (a, b), length(c)
    (; p, q) = (p = x * scale, q = y + offset)
    f = (u; w = 1) -> u * w + p
    g = let p = p
        v -> begin
            acc = 0
            for i in 1:v
                acc += f(i; w = q) - z
            end
            acc
        end
    end
    r = map(1:10) do i
        try
            i % 3 == 0 && error("x")
            g(i)
        catch err
            -1
        finally
            z += 1
        end
    end
    m = [i * j for i in 1:4, j in 1:3 if isodd(i + j)]
    d = Dict(k => v for (k, v) in zip(string.(1:5), r))
    h(x::T) where {T<:Real} = x isa Integer ? x : round(Int, x)
    local t = 0
    while t < 10
        t += h(t + 0.6)
        t > 5 && break
    end
    return (; r, m, d, t, kws = values(kws), flags)
end
"""
bench("lower_closures_kwargs"; setup = () -> (Meta.parseall(LOWER_SRC), Ref{Module}()), reset = st -> (st[2][] = fresh_module())) do (ex, mref)
    n = 0
    for _ in 1:4, e in ex.args
        e isa LineNumberNode && continue
        n += length(string(typeof(Meta.lower(mref[], e))))
    end
    n
end

const MACRO_SRC = """
Base.@kwdef struct Options
    tol::Float64 = 1e-8
    maxiter::Int = 100
    name::String = "solver"
    verbose::Bool = false
end
function kernel!(y, x, a)
    @assert length(x) == length(y) "length mismatch"
    @inbounds @simd for i in eachindex(x, y)
        y[i] = muladd(a, x[i], y[i])
    end
    @. y = 2y + sin(x)^2 - a
    r = r"(\\d+)-(\\w+)"
    s = String(b"bytes") * raw"C:\\path"
    v = @view x[2:end]
    t = @elapsed sum(v)
    @debug "done" t
    return y, s, r, Options(; name = "x")
end
"""
bench("macroexpand"; setup = () -> (Meta.parseall(MACRO_SRC), Ref{Module}()), reset = st -> (st[2][] = fresh_module())) do (ex, mref)
    n = 0
    for _ in 1:20, e in ex.args
        e isa LineNumberNode && continue
        n += length(string(typeof(macroexpand(mref[], e))))
    end
    n
end

# Inference and optimization of every entry point of a freshly defined workload
const SIGS = [
    (:simulate, Tuple{Int,Int}), (:word_freq, Tuple{String}), (:render_table, Tuple{Vector{NamedTuple{(:a, :b),Tuple{Int,String}}}}),
    (:matrix_stats, Tuple{Matrix{Float64}}), (:parse_records, Tuple{Vector{String}}),
    (:closures_and_generators, Tuple{Vector{Float64}}), (:try_paths, Tuple{Vector{Any}}),
    (:grid_sum, Tuple{Any}), (:run_all, Tuple{}),
]
bench("infer"; setup = () -> Ref{Module}(), reset = mref -> (mref[] = load(SRC)), seconds = 1.0) do mref
    n = 0
    for (name, sig) in SIGS
        n += length(Base.invokelatest(code_typed, latest(mref[], name), sig))
    end
    n
end

# Base's generic code inferred anew for a type that did not exist before: show, sort,
# broadcast, Dict, iteration. This is most of what a package's first use costs.
const NEWTYPE_SRC = """
struct Item
    id::Int
    w::Float64
    tag::Symbol
end
Base.isless(a::Item, b::Item) = isless(a.w, b.w)
function exercise(v::Vector{Item})
    d = Dict{Item,Int}()
    for (i, x) in enumerate(v)
        d[x] = i
    end
    s = sort(v)
    ws = map(x -> x.w, s) .* 2 .+ 1
    io = IOBuffer()
    show(io, MIME"text/plain"(), first(s, 2))
    show(io, d)
    return length(d), sum(ws), sum(x -> x.id, Iterators.filter(x -> x.tag === :a, v)), position(io)
end
"""
bench("infer_base_new_type"; setup = () -> Ref{Module}(), reset = mref -> (mref[] = load(NEWTYPE_SRC)), seconds = 1.0) do mref
    m = mref[]
    T = latest(m, :Item)
    length(Base.invokelatest(code_typed, latest(m, :exercise), Tuple{Vector{T}}))
end

# The first call of the workload: inference, optimization, LLVM and the JIT
bench("first_call"; setup = () -> Ref{Module}(), reset = mref -> (mref[] = load(SRC)), seconds = 1.0) do mref
    length(Base.invokelatest(latest(mref[], :run_all)))
end

bench("lower_large_function"; setup = () -> (Meta.parse(BIG_SRC), Ref{Module}()), reset = st -> (st[2][] = fresh_module())) do (ex, mref)
    length(string(typeof(Meta.lower(mref[], ex))))
end

bench("llvm_large_function"; setup = () -> Ref{Module}(), reset = mref -> (mref[] = load(BIG_SRC)), seconds = 1.0) do mref
    Base.invokelatest(latest(mref[], :big), 1.5, 2.5, 150)
end

# Inserting a method that 400 compiled callers depend on: method table insertion,
# backedge traversal and invalidation
const INVALIDATE_SRC = """
struct Thing end
g(x) = 1
""" * join(("caller$i(x) = g(x) + $i" for i in 1:400), "\n")
function prepare_invalidate()
    m = load(INVALIDATE_SRC)
    T = latest(m, :Thing)
    for i in 1:400
        Base.invokelatest(latest(m, Symbol("caller", i)), Base.invokelatest(T))
    end
    return m
end
bench("method_invalidate"; setup = () -> Ref{Module}(), reset = mref -> (mref[] = prepare_invalidate()), seconds = 1.0) do mref
    m = mref[]
    Core.eval(m, :(g(::Thing) = 2))
    m
end

end
