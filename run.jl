# Run benchmark categories in this process and append one JSON line per benchmark to --out.
#
#   julia run.jl --category=simd [--category=...] [--filter=REGEX] [--out=FILE]
#                [--seconds-scale=1.0] [--tag=key=value ...]
#
# bench.jl starts one process per category and arm; run this directly to try a category.
include(joinpath(@__DIR__, "driver", "Driver.jl"))
using .Driver

function main(args)
    categories = String[]
    filter = r""
    out = nothing
    scale = 1.0
    tags = Dict{String,Any}()
    for a in args
        k, v = occursin('=', a) ? split(a, '='; limit = 2) : (a, "")
        if k == "--category"
            push!(categories, v)
        elseif k == "--filter"
            filter = Regex(v)
        elseif k == "--out"
            out = String(v)
        elseif k == "--seconds-scale"
            scale = parse(Float64, v)
        elseif k == "--tag"
            tk, tv = split(v, '='; limit = 2)
            tags[tk] = something(tryparse(Int, tv), tv)
        else
            error("unknown argument $a")
        end
    end
    isempty(categories) && error("give at least one --category")
    for c in categories
        Driver.load_category(joinpath(@__DIR__, "benchmarks"), c)
    end
    tags["julia_version"] = string(VERSION)
    tags["threads"] = Threads.nthreads()
    if out === nothing
        Base.invokelatest(Driver.run_suite, devnull, filter, tags, scale)
    else
        open(io -> Base.invokelatest(Driver.run_suite, io, filter, tags, scale), out, "a")
    end
end

main(ARGS)
