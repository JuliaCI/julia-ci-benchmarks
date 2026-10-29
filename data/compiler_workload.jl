# A fixed body of ordinary Julia code for the compiler benchmarks. It is evaluated into a
# fresh module for every sample. Do not edit: changing it changes what is measured.

struct Particle{T<:Real}
    pos::NTuple{3,T}
    vel::NTuple{3,T}
    mass::T
end

mutable struct Stats
    n::Int
    mean::Float64
    m2::Float64
end
Stats() = Stats(0, 0.0, 0.0)

function push_stat!(s::Stats, x::Real)
    s.n += 1
    d = x - s.mean
    s.mean += d / s.n
    s.m2 += d * (x - s.mean)
    return s
end
variance(s::Stats) = s.n > 1 ? s.m2 / (s.n - 1) : NaN

kinetic(p::Particle) = 0.5 * p.mass * sum(abs2, p.vel)

function step(ps::Vector{Particle{T}}, dt::T) where {T}
    out = similar(ps)
    for (i, p) in enumerate(ps)
        v = p.vel .- dt .* p.pos ./ max(norm3(p.pos)^3, T(1e-3))
        out[i] = Particle(p.pos .+ dt .* v, v, p.mass)
    end
    return out
end
norm3(x::NTuple{3}) = sqrt(x[1]^2 + x[2]^2 + x[3]^2)

function simulate(n::Int, steps::Int)
    ps = [Particle((Float64(i), 1.0, -1.0), (0.0, 0.1 * i, 0.0), 1.0 + i / n) for i in 1:n]
    s = Stats()
    for _ in 1:steps
        ps = step(ps, 0.01)
        push_stat!(s, sum(kinetic, ps))
    end
    return s.mean, variance(s)
end

function word_freq(text::AbstractString)
    d = Dict{String,Int}()
    for w in eachsplit(lowercase(text), r"[^a-z]+"; keepempty = false)
        d[w] = get(d, w, 0) + 1
    end
    return sort!(collect(d); by = last, rev = true)
end

function render_table(rows::Vector{<:NamedTuple})
    isempty(rows) && return ""
    ks = keys(first(rows))
    widths = [maximum(r -> length(string(r[k])), rows; init = length(string(k))) for k in ks]
    io = IOBuffer()
    for (k, w) in zip(ks, widths)
        print(io, rpad(string(k), w + 2))
    end
    println(io)
    for r in rows
        for (k, w) in zip(ks, widths)
            print(io, rpad(string(r[k]), w + 2))
        end
        println(io)
    end
    return String(take!(io))
end

abstract type Expr2 end
struct Lit <: Expr2; v::Float64; end
struct Var <: Expr2; name::Symbol; end
struct BinOp <: Expr2; op::Symbol; a::Expr2; b::Expr2; end

evaluate(e::Lit, env) = e.v
evaluate(e::Var, env) = env[e.name]
function evaluate(e::BinOp, env)
    a, b = evaluate(e.a, env), evaluate(e.b, env)
    e.op === :+ ? a + b : e.op === :* ? a * b : e.op === :- ? a - b : a / b
end
build(d) = d == 0 ? (isodd(d) ? Var(:x) : Lit(2.0)) : BinOp(d % 2 == 0 ? :+ : :*, build(d - 1), d % 3 == 0 ? Var(:x) : Lit(d))

function matrix_stats(A::AbstractMatrix{T}) where {T<:Number}
    colmeans = vec(sum(A; dims = 1)) ./ size(A, 1)
    centered = A .- colmeans'
    cov = (centered' * centered) ./ (size(A, 1) - 1)
    return (; colmeans, trace = sum(cov[i, i] for i in axes(cov, 1)), maxabs = maximum(abs, cov))
end

function parse_records(lines)
    out = NamedTuple{(:id, :name, :score),Tuple{Int,String,Float64}}[]
    for l in lines
        parts = split(l, ',')
        length(parts) == 3 || continue
        id = tryparse(Int, parts[1])
        sc = tryparse(Float64, parts[3])
        (id === nothing || sc === nothing) && continue
        push!(out, (; id, name = String(strip(parts[2])), score = sc))
    end
    return out
end

function closures_and_generators(xs::Vector{T}) where {T}
    scale = length(xs) > 10 ? T(2) : T(1)
    f = x -> x * scale
    g = (x, y) -> x > y ? x - y : y - x
    a = sum(f, xs)
    b = maximum(g(x, y) for (x, y) in zip(xs, Iterators.reverse(xs)))
    c = count(x -> x > a / length(xs), xs)
    return a, b, c, [f(x)^2 for x in xs if x != 0]
end

function try_paths(xs)
    acc = 0
    for x in xs
        acc += try
            x isa Integer ? div(100, x) : round(Int, x)
        catch err
            err isa DivideError ? -1 : rethrow()
        finally
            acc += 0
        end
    end
    return acc
end

struct Grid{N}
    dims::NTuple{N,Int}
end
Base.length(g::Grid) = prod(g.dims)
Base.iterate(g::Grid, i = 1) = i > length(g) ? nothing : (CartesianIndices(g.dims)[i], i + 1)

function grid_sum(g::Grid{N}) where {N}
    s = 0
    for I in g
        s += sum(Tuple(I))
    end
    return s
end

function run_all()
    r1 = simulate(8, 4)
    r2 = word_freq("the quick brown fox jumps over the lazy dog the end")
    r3 = render_table([(; a = 1, b = "x"), (; a = 22, b = "yy")])
    r4 = evaluate(build(6), Dict(:x => 1.5))
    r5 = matrix_stats([1.0 2.0; 3.0 4.0; 5.0 7.0])
    r6 = parse_records(["1, a, 2.5", "2,b,x", "3,c,1e3"])
    r7 = closures_and_generators([1.0, 2.0, 3.0])
    r8 = closures_and_generators(collect(1:12))
    r9 = try_paths(Any[1, 0, 2.6, 5])
    r10 = grid_sum(Grid((3, 4, 2)))
    return (r1, length(r2), length(r3), r4, r5.trace, length(r6), r7[1], r8[3], r9, r10)
end
