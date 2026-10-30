# The benchmark driver. It loads nothing outside Base so that it runs on any julia build
# that boots, from 1.10 to the commit under test.
module Driver

include("json.jl")
using .BenchJSON
include("rng.jl")
include("counters.jl")

export bench, SplitMix, rand_f64, rand_f32, rand_int, shuffle_det!, carve, DEFAULT_SECONDS

struct Bench
    name::String
    kernel::Any
    setup::Any
    reset::Any
    ops::Int
    seconds::Float64
    min_threads::Int
    metric::String
    tolerance::Float64
end

const SUITE = Bench[]
const CATEGORY = Ref("")
const DEFAULT_SECONDS = 0.5
# Scales the sampling budget of benchmarks run interactively (tools/check_blocks.jl lowers it)
const INTERACTIVE_SCALE = Ref(1.0)
# Input files of the benchmarks, found the same way from a suite load and from the REPL
const DATA_DIR = joinpath(dirname(@__DIR__), "data")
const MIN_SAMPLES = 5
const MAX_SAMPLES = 300

"""
    bench(kernel, [setup]; reset, ops, seconds, min_threads, metric, tolerance)

Register the benchmark `<category>/<name of kernel>`. `args = setup()` is built once, a
tuple of arguments (or a single one), and one sample is one call `kernel(args...)`, never
a tuned number of calls, so a kernel should do at least ~20 µs of work. Without `setup`
the kernel takes no arguments. `reset(args...)`, when given, runs untimed before every
sample, for kernels that consume or mutate their input. `ops` is the number of work items
per call and only scales the reported time per item. `metric` is what a comparison
judges: `"cycles"` of the calling thread, `"time"` for work that leaves it (other
threads, parallel GC marking), or `"instructions"` of the calling thread for work whose
cycles follow heap placement too much to judge (the compiler on large inputs). `tolerance` overrides the comparison threshold.

Outside a suite run (the definitions pasted into a REPL) the benchmark runs right away.
"""
function bench(kernel, setup = () -> (); reset = nothing, ops::Int = 1,
               seconds::Real = DEFAULT_SECONDS, min_threads::Int = 1, metric::String = "cycles",
               tolerance::Real = 0.0)
    metric in ("cycles", "time", "instructions") || error("unknown metric $metric")
    name = string(nameof(kernel))
    interactive = isempty(CATEGORY[])
    full = interactive ? name : string(CATEGORY[], "/", name)
    b = Bench(full, kernel, setup, reset, ops, Float64(seconds), min_threads, metric, Float64(tolerance))
    # Outside a suite load (a block pasted into the REPL), run it right away
    interactive && return Base.invokelatest(run_interactive, b)
    any(x -> x.name == full, SUITE) && error("duplicate benchmark $full")
    push!(SUITE, b)
    return nothing
end

function run_interactive(b::Bench)
    backend = counter_backend()
    rec = run_one(b, backend, INTERACTIVE_SCALE[])
    rec.status == "ok" || (println(b.name, ": ", rec.status, " (", rec.error, ")"); return rec)
    parts = ["min $(format_ns(rec.min_ns)) (median $(format_ns(rec.median_ns)))"]
    b.ops > 1 && push!(parts, "$(format_ns(rec.min_ns / b.ops))/op")
    for (k, v) in sort!(collect(rec.counters); by = first)
        push!(parts, "$(format_count(minimum(v))) $k")
    end
    push!(parts, "$(rec.allocs) allocs", "$(length(rec.times_ns)) samples")
    println(b.name, ": ", join(parts, " · "))
    return rec
end

format_count(c) = c < 1e4 ? string(c) : c < 1e7 ? string(round(c / 1e3; digits = 1), "k") :
                  c < 1e10 ? string(round(c / 1e6; digits = 2), "M") : string(round(c / 1e9; digits = 2), "G")

function load_category(dir::String, category::String)
    CATEGORY[] = category
    Base.include(Main, joinpath(dir, category * ".jl"))
    CATEGORY[] = ""
    return nothing
end

median_of(v) = isempty(v) ? 0 : (s = sort(v); n = length(s); isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2)

function run_one(b::Bench, backend::CounterBackend, seconds_scale::Float64)
    if Threads.nthreads() < b.min_threads
        return (; status = "skipped", error = "needs $(b.min_threads) threads, have $(Threads.nthreads())")
    end
    args = b.setup()
    args isa Tuple || (args = (args,))
    budget = round(Int, b.seconds * seconds_scale * 1e9)
    # A function barrier, so the timed call is compiled for the kernel and its arguments
    times, allocs, bytes, gc_ns, counts = measure(b.kernel, b.reset, args, backend, budget)
    counters = Dict{String,Any}()
    for k in eachindex(counts)
        counters[backend.names[k]] = counts[k]
    end
    return (; status = "ok", error = nothing, ops = b.ops, times_ns = times,
            min_ns = minimum(times), median_ns = median_of(times),
            allocs = minimum(allocs), bytes = minimum(bytes), gc_ns = gc_ns,
            counters = counters, metric = b.metric, tolerance = b.tolerance == 0 ? nothing : b.tolerance)
end

function measure(kernel::F, reset::R, args::A, backend::CounterBackend, budget::Int) where {F,R,A}
    # Two untimed calls: the first compiles, the second settles caches and lazy state
    for _ in 1:2
        reset === nothing || reset(args...)
        Base.donotdelete(kernel(args...))
    end
    GC.gc()
    ncounters = backend.kind == :none ? 0 : length(backend.names)
    times = Int[]
    allocs = Int[]
    bytes = Int[]
    gc_ns = Int[]
    counts = [Int[] for _ in 1:ncounters]
    c0 = zeros(UInt64, max(ncounters, 1))
    c1 = zeros(UInt64, max(ncounters, 1))
    start = time_ns()
    while true
        reset === nothing || reset(args...)
        g0 = Base.gc_num()
        ncounters > 0 && read_counters!(backend, c0)
        t0 = time_ns()
        r = kernel(args...)
        t1 = time_ns()
        ncounters > 0 && read_counters!(backend, c1)
        g1 = Base.gc_num()
        Base.donotdelete(r)
        d = Base.GC_Diff(g1, g0)
        push!(times, Int(t1 - t0))
        push!(allocs, Int(Base.gc_alloc_count(d)))
        push!(bytes, Int(d.allocd))
        push!(gc_ns, Int(d.total_time))
        for k in 1:ncounters
            push!(counts[k], Int(c1[k] - c0[k]))
        end
        n = length(times)
        (n >= MAX_SAMPLES || (n >= MIN_SAMPLES && time_ns() - start >= budget)) && break
    end
    return times, allocs, bytes, gc_ns, counts
end

# Run every registered benchmark whose name matches `filter`, writing one JSON line per
# benchmark to `io` as soon as it finishes.
function run_suite(io::IO, filter::Regex, tags::Dict{String,Any}, seconds_scale::Float64)
    backend = counter_backend()
    for b in SUITE
        occursin(filter, b.name) || continue
        print(stderr, "  ", rpad(b.name, 44))
        rec = try
            run_one(b, backend, seconds_scale)
        catch err
            (; status = "error", error = sprint(showerror, err; context = :limit => true)[1:min(end, 2000)])
        end
        if rec.status == "ok"
            println(stderr, lpad(format_ns(rec.min_ns), 10), "  ", length(rec.times_ns), " samples")
        else
            println(stderr, rec.status, ": ", first(split(something(rec.error, ""), '\n')))
        end
        out = Dict{String,Any}(string(k) => v for (k, v) in pairs(rec))
        out["name"] = b.name
        out["category"] = first(split(b.name, '/'))
        out["counters_backend"] = string(backend.kind)
        merge!(out, tags)
        println(io, replace(json_write(out), r"\n\s*" => ""))
        flush(io)
        GC.gc()
    end
end

function format_ns(t)
    t < 1e3 && return string(round(t; digits = 1), " ns")
    t < 1e6 && return string(round(t / 1e3; digits = 2), " µs")
    t < 1e9 && return string(round(t / 1e6; digits = 2), " ms")
    return string(round(t / 1e9; digits = 2), " s")
end

end # module
