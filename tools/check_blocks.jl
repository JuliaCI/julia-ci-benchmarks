# Check that every benchmark block runs on its own, as when pasted into a REPL: for each
# category file, evaluate the shared part (if any) and then one block into a fresh module
# that has only the driver, which runs the benchmark immediately.
#
#   julia -t 4 tools/check_blocks.jl [category ...]
#
# A block is the top-level code after the previous `bench` call up to and including its
# own; a failure means the block depends on something defined in another block.
include(joinpath(@__DIR__, "..", "driver", "Driver.jl"))
using .Driver

Driver.INTERACTIVE_SCALE[] = 0.02

is_bench(ex) = occursin("bench(", string(ex))

function blocks(path)
    body = Meta.parseall(read(path, String)).args
    mod = only(filter(e -> Meta.isexpr(e, :module), body))
    stmts = filter(e -> !(e isa LineNumberNode), mod.args[end].args)  # newer versions prepend a version
    stmts = filter(e -> !(Meta.isexpr(e, :using)), stmts)
    src = read(path, String)
    shared = Any[]
    if occursin("# End of the shared part.", src)
        # Statements before the first bench call whose names the shared part defines
        marker = "# Shared by the benchmarks below: paste this part first to run one of them in the REPL.\n"
        pre = Meta.parseall(split(split(src, marker)[2], "# End of the shared part.")[1])
        shared = filter(e -> !(e isa LineNumberNode), pre.args)
        shared_strs = Set(string.(shared))
        stmts = filter(e -> !(string(e) in shared_strs), stmts)
    end
    out = Vector{Any}[]
    cur = Any[]
    for e in stmts
        push!(cur, e)
        if is_bench(e)
            push!(out, cur)
            cur = Any[]
        end
    end
    return shared, out
end

function check(category)
    path = joinpath(@__DIR__, "..", "benchmarks", category * ".jl")
    shared, bs = blocks(path)
    failures = String[]
    for (i, b) in enumerate(bs)
        # Bound in Main as a REPL paste would be, so that serialization finds its types
        m = Core.eval(Main, :(module $(Symbol("Paste_", category, "_", i)) end))
        Core.eval(m, :(using Main.Driver))
        Core.eval(m, :(using Serialization))
        try
            for e in shared
                Core.eval(m, e)
            end
            redirect_stdout(devnull) do
                for e in b
                    rec = Core.eval(m, e)
                    rec isa NamedTuple && haskey(rec, :status) || continue
                    rec.status == "ok" || rec.status == "skipped" || error(rec.error)
                end
            end
        catch err
            push!(failures, sprint(showerror, err)[1:min(end, 300)])
            println("  FAIL in block ending with: ", first(split(string(b[end]), '\n')))
            println("       ", failures[end])
        end
    end
    println(rpad(category, 12), length(bs), " blocks, ", length(failures), " failed")
    return isempty(failures)
end

cats = isempty(ARGS) ? first.(splitext.(sort(readdir(joinpath(@__DIR__, "..", "benchmarks"))))) : ARGS
ok = all([check(c) for c in cats])
exit(ok ? 0 : 1)
