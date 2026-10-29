# Judge and report results written by bench.jl.
#
#   julia compare.jl RESULTS_DIR [--base=base] [--head=head] [--perf-url=URL]
#
# With two arms, compares head against base and writes RESULTS_DIR/report.md and
# RESULTS_DIR/compare.json. With one arm, writes a summary report. Exit code:
#   0  no differences (with one arm: a complete run)
#   3  improvements only
#   1  regressions, whether or not the run is complete
#   2  incomplete, no regressions: a benchmark raised an error or a process failed
# The report lists every incomplete benchmark under "Incomplete", including those with
# missing blocks or blocks that differ between arms; a skipped benchmark (`min_threads`)
# is not incomplete.
#
# Per benchmark, arm and block the value is the minimum over samples of the benchmark's
# metric: cycles of the calling thread, or wall time for work that leaves it. A benchmark
# whose records disagree on the unit (counters missing on some arm or block) is judged on
# time throughout. A change is reported only when it is robust: every block's ratio
# exceeds the tolerance, the head and base values do not overlap at all, and the
# difference clears an absolute floor.
# Instruction counts are exact from process to process, so a change in them is reported
# separately: the generated code or the work done changed, whether or not it is slower.
include(joinpath(@__DIR__, "driver", "json.jl"))
using .BenchJSON
using Printf

const DEFAULT_TOL = Dict("cycles" => 0.03, "time" => 0.06)
const FLOOR = Dict("cycles" => 5_000, "time" => 2_000)  # cycles, ns
const INSTR_TOL = 0.005
const SUITE_TOL = 0.015

struct Cell
    value::Float64
    time::Float64
    instructions::Union{Nothing,Float64}
    allocs::Int
    bytes::Int
end

function load(dir)
    recs = json_parse(read(joinpath(dir, "results.json"), String))
    meta = json_parse(read(joinpath(dir, "results-meta.json"), String))
    return recs, meta
end

function metric_of(r)
    m = r["metric"]
    m == "cycles" && haskey(r["counters"], "cycles") ? "cycles" : "time"
end

function cell(r, m)
    c = r["counters"]
    v = m == "cycles" ? minimum(c["cycles"]) : r["min_ns"]
    ins = haskey(c, "instructions") ? Float64(minimum(c["instructions"])) : nothing
    return Cell(v, r["min_ns"], ins, r["allocs"], r["bytes"])
end

# cells[name][arm][block] => Cell. A benchmark is judged in one unit: when its records
# disagree (cycles on one arm or block, time on another), time for all of them.
function collect_cells(recs)
    units = Dict{String,Set{String}}()
    for r in recs
        r["status"] == "ok" && push!(get!(Set{String}, units, r["name"]), metric_of(r))
    end
    cells = Dict{String,Dict{String,Dict{Int,Cell}}}()
    info = Dict{String,Any}()
    errors = Dict{String,Dict{String,String}}()
    skipped = Dict{String,Dict{String,String}}()
    for r in recs
        name, arm = r["name"], r["arm"]
        if r["status"] != "ok"
            get!(Dict{String,String}, r["status"] == "skipped" ? skipped : errors, name)[arm] = something(r["error"], r["status"])
            continue
        end
        m = length(units[name]) == 1 ? only(units[name]) : "time"
        get!(() -> Dict{String,Dict{Int,Cell}}(), cells, name)
        get!(Dict{Int,Cell}, cells[name], arm)[r["block"]] = cell(r, m)
        info[name] = (; category = r["category"], metric = m, ops = r["ops"], mixed_units = length(units[name]) > 1,
                      tolerance = something(r["tolerance"], DEFAULT_TOL[m]))
    end
    return cells, info, errors, skipped
end

# Benchmarks whose comparison rests on less than the run promised: an error on some arm,
# or blocks that differ between arms or fall short of the blocks run. Blocks lost with a
# failed process are reported once, as that process.
function incompleteness(cells, errors, meta, arms)
    nblocks = meta["settings"]["blocks"]
    failed = Set((f["category"], f["arm"], f["block"]) for f in meta["failures"])
    out = Dict{String,Vector{String}}()
    for (name, by_arm) in errors, (arm, err) in by_arm
        push!(get!(Vector{String}, out, name), "error on `$arm`: $(first(split(err, '\n')))")
    end
    for (name, by_arm) in cells
        got = Dict(a => sort!(collect(keys(get(by_arm, a, Dict{Int,Cell}())))) for a in arms)
        all(a -> got[a] == 1:nblocks, arms) && continue
        cat = first(split(name, '/'))
        lost = [(a, k) for a in arms for k in 1:nblocks if !(k in got[a])]
        haskey(errors, name) || all(((a, k),) -> (cat, a, k) in failed, lost) && continue
        push!(get!(Vector{String}, out, name), "blocks " * join(("`$a` $(got[a])" for a in arms), ", ") * " of $nblocks")
    end
    return out
end

function incomplete_report(io, incomplete, skipped, meta)
    if !isempty(incomplete) || !isempty(meta["failures"])
        println(io, "\n#### Incomplete\n")
        for f in meta["failures"]
            println(io, "- process for `$(f["category"])` on `$(f["arm"])`, block $(f["block"]) failed: $(f["error"])")
        end
        for (name, why) in sort!(collect(incomplete))
            println(io, "- `$name`: ", join(why, "; "))
        end
    end
    if !isempty(skipped)
        println(io, "\nSkipped: ", join(("`$name` ($(first(values(by_arm))))" for (name, by_arm) in sort!(collect(skipped))), ", "))
    end
end

fmt_time(t) = t < 1e3 ? @sprintf("%.0f ns", t) : t < 1e6 ? @sprintf("%.1f µs", t / 1e3) :
              t < 1e9 ? @sprintf("%.2f ms", t / 1e6) : @sprintf("%.2f s", t / 1e9)
fmt_count(c) = c < 1e4 ? @sprintf("%.0f", c) : c < 1e7 ? @sprintf("%.1fk", c / 1e3) :
               c < 1e10 ? @sprintf("%.2fM", c / 1e6) : @sprintf("%.2fG", c / 1e9)
fmt_value(v, metric) = metric == "cycles" ? fmt_count(v) * " cyc" : fmt_time(v)
fmt_pct(r) = @sprintf("%+.1f%%", 100 * (r - 1))
geomean(xs) = exp(sum(log, xs) / length(xs))

function judge(cells, info, base, head)
    rows = Any[]
    for name in sort!(collect(keys(cells)))
        arms = cells[name]
        (haskey(arms, base) && haskey(arms, head)) || continue
        blocks = sort!(collect(intersect(keys(arms[base]), keys(arms[head]))))
        isempty(blocks) && continue
        inf = info[name]
        b = [arms[base][k].value for k in blocks]
        h = [arms[head][k].value for k in blocks]
        ratios = h ./ b
        tol = inf.tolerance
        floor_ok = abs(minimum(h) - minimum(b)) >= FLOOR[inf.metric]
        verdict = if all(>(1 + tol), ratios) && minimum(h) > maximum(b) && floor_ok
            "regression"
        elseif all(<(1 / (1 + tol)), ratios) && maximum(h) < minimum(b) && floor_ok
            "improvement"
        else
            "same"
        end
        bi = [arms[base][k].instructions for k in blocks]
        hi = [arms[head][k].instructions for k in blocks]
        # Only the calling thread is counted, which says little about work spread over threads
        instr_ratio = inf.metric != "cycles" || any(isnothing, bi) || any(isnothing, hi) ? nothing :
                      minimum(hi) / minimum(bi)
        # Instruction counts repeat exactly unless GC or system calls do work on the calling
        # thread; a benchmark whose own blocks disagree has no usable instruction signal
        spread(v) = (maximum(v) - minimum(v)) / minimum(v)
        instr_stable = instr_ratio !== nothing && spread(bi) <= INSTR_TOL / 2 && spread(hi) <= INSTR_TOL / 2
        instr_changed = instr_stable && abs(instr_ratio - 1) > INSTR_TOL &&
                        all(k -> abs(hi[k] / bi[k] - 1) > INSTR_TOL, eachindex(blocks))
        ab = minimum(arms[base][k].allocs for k in blocks)
        ah = minimum(arms[head][k].allocs for k in blocks)
        push!(rows, (; name, category = inf.category, metric = inf.metric, mixed_units = inf.mixed_units, tolerance = tol, verdict,
                     base = minimum(b), head = minimum(h), ratio = minimum(h) / minimum(b), ratios,
                     base_time = minimum(arms[base][k].time for k in blocks),
                     head_time = minimum(arms[head][k].time for k in blocks),
                     instr_ratio, instr_changed, allocs_base = ab, allocs_head = ah, blocks = length(blocks)))
    end
    return rows
end

function suite_verdicts(rows)
    out = Any[]
    for cat in sort!(unique(r.category for r in rows))
        rs = filter(r -> r.category == cat, rows)
        nb = minimum(r -> length(r.ratios), rs)
        per_block = [geomean([r.ratios[k] for r in rs]) for k in 1:nb]
        verdict = all(>(1 + SUITE_TOL), per_block) ? "regression" :
                  all(<(1 / (1 + SUITE_TOL)), per_block) ? "improvement" : "same"
        push!(out, (; category = cat, geomean = geomean([r.ratio for r in rs]), per_block, verdict, n = length(rs)))
    end
    return out
end

arm_line(meta, a) = (i = meta["arms"][a]; "`$a`: julia $(i["version"]) `$(first(i["commit"], 10))` (LLVM $(i["llvm"]))")

function compare_report(io, rows, suites, incomplete, skipped, meta, base, head, perf_url)
    nreg = count(r -> r.verdict == "regression", rows)
    nimp = count(r -> r.verdict == "improvement", rows)
    sreg = count(s -> s.verdict == "regression", suites)
    ninc = length(incomplete) + length(meta["failures"])
    icon = nreg + sreg > 0 ? "🔴" : ninc > 0 ? "⚠️" : nimp > 0 ? "🟢" : "⚪"
    println(io, "### $icon Runtime benchmarks: $nreg regressions, $nimp improvements",
            ninc > 0 ? ", incomplete (see below)" : "")
    println(io)
    println(io, "$(length(rows)) benchmarks · $(arm_line(meta, base)) → $(arm_line(meta, head)) · ",
            "$(meta["settings"]["blocks"]) blocks · $(meta["machine"]["cpu"])")
    perf_url === nothing || println(io, "\n[All measurements of this build on perf.julialang.org]($perf_url)")
    println(io)
    changed = filter(r -> r.verdict != "same", rows)
    if !isempty(changed)
        println(io, "| | Benchmark | Metric | Base | Head | Change | Per block | Instructions |")
        println(io, "|:-:|:--|:--|--:|--:|--:|:--|--:|")
        for r in sort(changed; by = r -> -abs(log(r.ratio)))
            mark = r.verdict == "regression" ? "🔴" : "🟢"
            instr = r.instr_ratio === nothing ? "" : fmt_pct(r.instr_ratio)
            println(io, "| $mark | `$(r.name)` | $(r.metric) | $(fmt_value(r.base, r.metric)) | $(fmt_value(r.head, r.metric)) | ",
                    "$(fmt_pct(r.ratio)) | $(join(fmt_pct.(r.ratios), " ")) | $instr |")
        end
        println(io)
    end
    if any(s -> s.verdict != "same", suites)
        println(io, "**Category geometric means:** ",
                join(("$(s.verdict == "regression" ? "🔴" : "🟢") $(s.category) $(fmt_pct(s.geomean))" for s in suites if s.verdict != "same"), ", "))
        println(io)
    end
    instr = filter(r -> r.instr_changed && r.verdict == "same", rows)
    if !isempty(instr)
        println(io, "<details><summary>$(length(instr)) benchmarks retire a different number of instructions without a robust time change</summary>\n")
        println(io, "| Benchmark | Instructions | $(rows[1].metric == "cycles" ? "Cycles" : "Metric") |")
        println(io, "|:--|--:|--:|")
        for r in sort(instr; by = r -> -abs(log(r.instr_ratio)))
            println(io, "| `$(r.name)` | $(fmt_pct(r.instr_ratio)) | $(fmt_pct(r.ratio)) |")
        end
        println(io, "\n</details>\n")
    end
    mixed = filter(r -> r.mixed_units, rows)
    if !isempty(mixed)
        println(io, "Judged on time because cycles were missing on some arm or block: ",
                join(("`$(r.name)`" for r in mixed), ", "), "\n")
    end
    println(io, "<details><summary>All benchmarks</summary>\n")
    println(io, "| Benchmark | Metric | Base | Head | Change | Per block | Instr. | Time (head) | Allocs |")
    println(io, "|:--|:--|--:|--:|--:|:--|--:|--:|--:|")
    for r in rows
        instr = r.instr_ratio === nothing ? "" : fmt_pct(r.instr_ratio)
        allocs = r.allocs_base == r.allocs_head ? string(r.allocs_head) : "$(r.allocs_base) → $(r.allocs_head)"
        println(io, "| `$(r.name)` | $(r.metric) | $(fmt_value(r.base, r.metric)) | $(fmt_value(r.head, r.metric)) | $(fmt_pct(r.ratio)) | ",
                "$(join(fmt_pct.(r.ratios), " ")) | $instr | $(fmt_time(r.head_time)) | $allocs |")
    end
    println(io, "\n| Category | Geomean | Per block |")
    println(io, "|:--|--:|:--|")
    for s in suites
        println(io, "| $(s.category) | $(fmt_pct(s.geomean)) | $(join(fmt_pct.(s.per_block), " ")) |")
    end
    println(io, "\n</details>")
    incomplete_report(io, incomplete, skipped, meta)
    println(io, "\n<sub>A change counts when every block agrees beyond the tolerance (cycles 3%, time 6% unless the benchmark sets its own) ",
            "and the arms do not overlap. Instruction counts are exact across processes; a change there means the code or work changed.</sub>")
end

function summary_report(io, cells, info, incomplete, skipped, meta, arm)
    i = meta["arms"][arm]
    println(io, "### Runtime benchmarks: julia $(i["version"]) `$(first(i["commit"], 10))`")
    println(io)
    println(io, "$(length(cells)) benchmarks · $(meta["settings"]["blocks"]) blocks · $(meta["machine"]["cpu"]) · LLVM $(i["llvm"])")
    println(io)
    println(io, "| Benchmark | Time | Per op | Cycles | Instructions | IPC | Allocs |")
    println(io, "|:--|--:|--:|--:|--:|--:|--:|")
    for name in sort!(collect(keys(cells)))
        haskey(cells[name], arm) || continue
        cs = collect(values(cells[name][arm]))
        t = minimum(c -> c.time, cs)
        ins = cs[1].instructions
        cyc = info[name].metric == "cycles" ? minimum(c -> c.value, cs) : nothing
        ipc = ins === nothing || cyc === nothing ? "" : @sprintf("%.2f", ins / cyc)
        println(io, "| `$name` | $(fmt_time(t)) | $(fmt_time(t / info[name].ops)) | ",
                "$(cyc === nothing ? "" : fmt_count(cyc)) | $(ins === nothing ? "" : fmt_count(ins)) | $ipc | $(cs[1].allocs) |")
    end
    incomplete_report(io, incomplete, skipped, meta)
end

function main(args)
    dir = nothing
    base, head = "base", "head"
    perf_url = nothing
    for a in args
        if startswith(a, "--base=")
            base = a[8:end]
        elseif startswith(a, "--head=")
            head = a[8:end]
        elseif startswith(a, "--perf-url=")
            perf_url = a[12:end]
        else
            dir = a
        end
    end
    dir === nothing && error("usage: julia compare.jl RESULTS_DIR [--base=NAME] [--head=NAME]")
    recs, meta = load(dir)
    cells, info, errors, skipped = collect_cells(recs)
    arms = meta["arm_order"]
    # Errors and failed processes make the run incomplete; a regression still takes precedence
    broken = !isempty(errors) || !isempty(meta["failures"])
    if length(arms) == 1
        incomplete = incompleteness(cells, errors, meta, arms)
        open(io -> summary_report(io, cells, info, incomplete, skipped, meta, arms[1]), joinpath(dir, "report.md"), "w")
        return broken ? 2 : 0
    end
    incomplete = incompleteness(cells, errors, meta, [base, head])
    rows = judge(cells, info, base, head)
    suites = suite_verdicts(rows)
    open(io -> compare_report(io, rows, suites, incomplete, skipped, meta, base, head, perf_url), joinpath(dir, "report.md"), "w")
    tojson(r) = Dict{String,Any}(string(k) => v for (k, v) in pairs(r))
    write(joinpath(dir, "compare.json"), json_write(Dict(
        "base" => base, "head" => head, "benchmarks" => tojson.(rows), "categories" => tojson.(suites),
        "errors" => errors, "skipped" => skipped, "incomplete" => incomplete, "failures" => meta["failures"])))
    nreg = count(r -> r.verdict == "regression", rows) + count(s -> s.verdict == "regression", suites)
    nimp = count(r -> r.verdict == "improvement", rows)
    return nreg > 0 ? 1 : broken ? 2 : nimp > 0 ? 3 : 0
end

exit(main(ARGS))
