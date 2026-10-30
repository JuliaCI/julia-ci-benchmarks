# What to measure again, with a control build, after a comparison found regressions.
#
#   julia tools/flagged.jl RESULTS_DIR
#
# Prints the categories to run on the first line and a --filter regex on the second: every
# benchmark flagged as a regression, and every benchmark of a category whose geometric
# mean was (the mean needs the whole category). Prints nothing without regressions.
include(joinpath(@__DIR__, "..", "driver", "json.jl"))
using .BenchJSON

c = json_parse(read(joinpath(ARGS[1], "compare.json"), String))
names = [b["name"] for b in c["benchmarks"] if b["verdict"] == "regression"]
cats = [s["category"] for s in c["categories"] if s["verdict"] == "regression"]
if !isempty(names) || !isempty(cats)
    println(join(sort!(unique([cats; first.(split.(names, '/'))])), ' '))
    println("^(", join([[string(c, "/") for c in cats]; [string(n, "\$") for n in names]], '|'), ")")
end
