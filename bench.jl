# Run the suite for one or more julia builds ("arms") and collect the results.
#
#   julia bench.jl --arm=head=/path/to/julia [--arm=base=/path/to/julia]
#                  [--blocks=3] [--out=results] [--category=NAME ...] [--filter=REGEX]
#                  [--threads=4] [--seconds-scale=1.0] [--timeout=900]
#
# Each (block, category, arm) is a fresh process: process-level state (heap layout, code
# placement) is the largest source of noise, so the budget goes to more processes rather
# than longer sampling. Arms alternate order between consecutive runs (ABBA) so drift
# lands on both alike. The orchestrating julia only spawns processes; it can be any build.
#
# Writes OUT/results.json (one record per benchmark, arm and block) and
# OUT/results-meta.json (builds, machine, settings).
include(joinpath(@__DIR__, "driver", "json.jl"))
using .BenchJSON

const CATEGORIES = ["scalar", "simd", "array", "collections", "strings", "io", "dispatch",
                    "types", "runtime", "gc", "compiler", "inference", "apps", "parallel"]
# Categories that need more than one thread
const THREADED = Set(["parallel"])

function parse_args(args)
    o = Dict{String,Any}("arms" => Pair{String,String}[], "blocks" => 3, "out" => "results",
                         "categories" => String[], "filter" => "", "threads" => 4,
                         "seconds-scale" => 1.0, "timeout" => 900)
    for a in args
        k, v = occursin('=', a) ? split(a, '='; limit = 2) : (a, "")
        k = lstrip(k, '-')
        if k == "arm"
            name, path = split(v, '='; limit = 2)
            push!(o["arms"], String(name) => String(path))
        elseif k == "category"
            push!(o["categories"], String(v))
        elseif k in ("blocks", "threads", "timeout")
            o[k] = parse(Int, v)
        elseif k == "seconds-scale"
            o[k] = parse(Float64, v)
        elseif k in ("out", "filter")
            o[k] = String(v)
        else
            error("unknown argument $a")
        end
    end
    isempty(o["arms"]) && error("give at least one --arm=name=/path/to/julia")
    isempty(o["categories"]) && append!(o["categories"], CATEGORIES)
    return o
end

function arm_info(julia)
    code = """
    print(VERSION, "\\n", something(Base.GIT_VERSION_INFO.commit, ""), "\\n",
          Base.GIT_VERSION_INFO.date_string, "\\n", Sys.CPU_NAME, "\\n", Base.libllvm_version)
    """
    lines = readlines(`$julia --startup-file=no -e $code`)
    return Dict{String,Any}("path" => julia, "version" => lines[1], "commit" => lines[2],
                            "date" => lines[3], "cpu_target" => lines[4], "llvm" => lines[5])
end

function machine_info()
    info = Dict{String,Any}("os" => string(Sys.KERNEL), "arch" => string(Sys.ARCH),
                            "cpu" => Sys.cpu_info()[1].model, "ncpu" => Sys.CPU_THREADS,
                            "memory_gib" => round(Sys.total_memory() / 2^30; digits = 1),
                            "hostname" => gethostname())
    if Sys.isapple()
        for (k, key) in (("perf_cores", "hw.perflevel0.physicalcpu"), ("efficiency_cores", "hw.perflevel1.physicalcpu"),
                         ("model", "hw.model"))
            v = try readchomp(`sysctl -n $key`) catch; "" end
            isempty(v) || (info[k] = something(tryparse(Int, v), v))
        end
    end
    return info
end

function git_commit(dir)
    try
        readchomp(pipeline(`git -C $dir rev-parse HEAD`; stderr = devnull))
    catch
        ""
    end
end

# Run one process, killing it after `timeout` seconds; returns (ok, message)
function run_process(cmd, log, timeout)
    p = run(pipeline(cmd; stdout = log, stderr = log); wait = false)
    timer = Timer(timeout) do _
        process_running(p) && kill(p)
    end
    wait(p)
    close(timer)
    success(p) && return true, ""
    return false, p.termsignal != 0 ? "killed by signal $(p.termsignal) (timeout $(timeout)s?)" : "exit code $(p.exitcode)"
end

function main(args)
    o = parse_args(args)
    out = mkpath(o["out"])
    lines_path = joinpath(out, "results.jsonl")
    rm(lines_path; force = true)
    logdir = mkpath(joinpath(out, "logs"))
    arms = o["arms"]
    meta = Dict{String,Any}(
        "arms" => Dict(name => arm_info(path) for (name, path) in arms),
        "arm_order" => first.(arms),
        "machine" => machine_info(),
        "suite_commit" => git_commit(@__DIR__),
        "settings" => Dict("blocks" => o["blocks"], "threads" => o["threads"], "categories" => o["categories"],
                           "filter" => o["filter"], "seconds_scale" => o["seconds-scale"]),
        "started" => string(Base.Libc.strftime("%Y-%m-%dT%H:%M:%S", time())),
        "buildkite" => Dict(k => ENV[k] for k in ("BUILDKITE_BUILD_URL", "BUILDKITE_JOB_ID", "BUILDKITE_PIPELINE_SLUG",
                                                  "BUILDKITE_BRANCH", "BUILDKITE_PULL_REQUEST", "BUILDKITE_AGENT_NAME")
                            if haskey(ENV, k)),
    )
    for (name, info) in meta["arms"]
        println(stderr, "arm $name: julia $(info["version"]) $(first(info["commit"], 10)) ($(info["path"]))")
    end
    failures = Any[]
    order = 0
    t_start = time()
    for block in 1:o["blocks"], (ci, cat) in enumerate(o["categories"])
        block_arms = isodd(block + ci) ? arms : reverse(arms)
        for (name, julia) in block_arms
            order += 1
            threads = cat in THREADED ? o["threads"] : 1
            println(stderr, "[block $block] $cat on $name ($(round(Int, time() - t_start)) s)")
            cmd = `$julia --startup-file=no --history-file=no -t $threads $(joinpath(@__DIR__, "run.jl"))
                   --category=$cat --out=$lines_path --seconds-scale=$(o["seconds-scale"])
                   --tag=arm=$name --tag=block=$block --tag=order=$order`
            isempty(o["filter"]) || (cmd = `$cmd --filter=$(o["filter"])`)
            cmd = addenv(cmd, "JULIA_LOAD_PATH" => "@stdlib", "JULIA_PROJECT" => nothing)
            logfile = joinpath(logdir, "$cat-$name-$block.log")
            ok, msg = open(log -> run_process(cmd, log, o["timeout"]), logfile, "w")
            print(stderr, read(logfile, String))
            if !ok
                println(stderr, "  FAILED: $msg")
                push!(failures, Dict("category" => cat, "arm" => name, "block" => block, "error" => msg,
                                     "log" => relpath(logfile, out)))
            end
        end
    end
    meta["failures"] = failures
    meta["duration_s"] = round(time() - t_start; digits = 1)
    records = isfile(lines_path) ? readlines(lines_path) : String[]
    open(joinpath(out, "results.json"), "w") do io
        println(io, "[")
        join(io, records, ",\n")
        println(io, "\n]")
    end
    rm(lines_path)
    write(joinpath(out, "results-meta.json"), json_write(meta))
    println(stderr, "$(length(records)) records in $(meta["duration_s"]) s, $(length(failures)) failed processes")
    return isempty(failures) ? 0 : 2
end

exit(main(ARGS))
