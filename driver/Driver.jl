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
const MIN_SAMPLES = 5
const MAX_SAMPLES = 300

"""
    bench(kernel, name; setup, reset, ops, seconds, min_threads, counters, tolerance)

Register `kernel(state)` as benchmark `<category>/<name>`. `state = setup()` is built once.
`reset(state)`, when given, runs untimed before every sample, for kernels that consume or
mutate their input. One sample is one call of `kernel`, never a tuned number of calls, so
a kernel should do at least ~20 µs of work. `ops` is the number of work items per call and
only scales the reported time per item. `metric` is what a comparison judges: `"cycles"`
of the calling thread, or `"time"` for work that leaves it (other threads, parallel GC
marking). `tolerance` overrides the comparison threshold for that metric.
"""
function bench(kernel, name::String; setup = () -> nothing, reset = nothing, ops::Int = 1,
               seconds::Real = DEFAULT_SECONDS, min_threads::Int = 1, metric::String = "cycles",
               tolerance::Real = 0.0)
    full = string(CATEGORY[], "/", name)
    any(b -> b.name == full, SUITE) && error("duplicate benchmark $full")
    metric in ("cycles", "time") || error("unknown metric $metric")
    push!(SUITE, Bench(full, kernel, setup, reset, ops, Float64(seconds), min_threads, metric,
                       Float64(tolerance)))
    return nothing
end

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
    state = b.setup()
    # Two untimed calls: the first compiles, the second settles caches and lazy state
    for _ in 1:2
        b.reset === nothing || b.reset(state)
        Base.donotdelete(b.kernel(state))
    end
    GC.gc()
    use_counters = backend.kind != :none
    ncounters = use_counters ? length(backend.names) : 0
    times = Int[]
    allocs = Int[]
    bytes = Int[]
    gc_ns = Int[]
    counts = [Int[] for _ in 1:ncounters]
    c0 = zeros(UInt64, max(ncounters, 1))
    c1 = zeros(UInt64, max(ncounters, 1))
    budget = round(Int, b.seconds * seconds_scale * 1e9)
    start = time_ns()
    while true
        b.reset === nothing || b.reset(state)
        g0 = Base.gc_num()
        use_counters && read_counters!(backend, c0)
        t0 = time_ns()
        r = b.kernel(state)
        t1 = time_ns()
        use_counters && read_counters!(backend, c1)
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
        elapsed = time_ns() - start
        (n >= MAX_SAMPLES || (n >= MIN_SAMPLES && elapsed >= budget)) && break
    end
    counters = Dict{String,Any}()
    for k in 1:ncounters
        counters[backend.names[k]] = counts[k]
    end
    return (; status = "ok", error = nothing, ops = b.ops, times_ns = times,
            min_ns = minimum(times), median_ns = median_of(times),
            allocs = minimum(allocs), bytes = minimum(bytes), gc_ns = gc_ns,
            counters = counters, metric = b.metric, tolerance = b.tolerance == 0 ? nothing : b.tolerance)
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
