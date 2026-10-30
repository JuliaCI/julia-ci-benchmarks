# julia-ci-benchmarks

A runtime benchmark suite for Julia itself, small enough to run on every commit to
`master` and on pull requests, on the same macOS aarch64 machine as the TTFX job. It
complements [BaseBenchmarks.jl](https://github.com/JuliaCI/BaseBenchmarks.jl): instead of
thousands of overlapping cases, it has 145 benchmarks, each chosen to cover a code path
the others do not. The suite targets what changes when *Julia* changes: codegen and
aliasing information, LLVM upgrades, inference, lowering, the runtime, the GC and the
scheduler. Library performance (LinearAlgebra, SparseArrays, ...) is out of scope.

```sh
# one build, absolute numbers
julia bench.jl --arm=head=/path/to/julia --out=results
julia compare.jl results            # results/report.md

# two builds, interleaved comparison
julia bench.jl --arm=base=/path/to/old/julia --arm=head=/path/to/new/julia --out=results
julia compare.jl results            # report.md, compare.json; exit 1 on a regression

# try one category in this process
julia run.jl --category=simd
```

Nothing outside Base and the stdlibs is loaded, so the suite runs on any build that
boots, from 1.10 to the commit under test, with no package installation.

## Design

These choices come from measurements on an M4 Pro, which are summarized below.

**Hardware counters are the primary metric.** The driver reads the calling thread's
cycles and retired instructions around every sample, through `thread_selfcounts`, which
needs no privileges on macOS. Across eight fresh processes running the same build, the
measured spread was:

| | wall time | cycles | instructions |
|---|---|---|---|
| range / median | 6.5–12% | 0.4–4% | 0.000% |

Wall time picks up frequency scaling and moves between performance and efficiency cores;
cycles do not. Instruction counts are exact from process to process, even for a 150 ns
kernel. So single-threaded benchmarks are judged on **cycles**. A change in
**instructions** is reported separately, as "the generated code or the work changed",
whether or not it costs time. Benchmarks whose work leaves the calling thread (the
scheduler, parallel GC marking, collections) are judged on wall time with a wider
tolerance. With root, the driver also records branch misses and L1D misses through kperf
(`CIBENCH_COUNTERS=kperf`); without root it falls back to cycles and instructions.

**Processes, not samples.** Every driver tested showed a 6–11% spread in the minimum
across fresh processes, however long each process sampled. Heap layout, ASLR and code
placement are fixed per process. So the budget buys more processes: each
(block, category, arm) is its own process, the arms alternate order (ABBA), and a
verdict needs every block to agree.

**No tuning; kernels are sized instead.** One sample is one call of the kernel, and every
kernel does at least ~20 µs of work. Tuning, i.e. packing many calls of a tiny function
into one sample, does not make such a benchmark usable: BenchmarkTools with 5 s of
tuned sampling still had an 11.5% process-to-process spread on a 150 ns kernel. It also
costs time, and it can choose a different evaluation count for each arm. A realistic
batch of inputs avoids both problems.

**Inputs defeat the branch predictor.** The M4's predictor learns a repeated random
sequence of up to ~4K branches perfectly. A branchy loop over 1000 random inputs costs the
same as over sorted inputs, and only from ~16K inputs does the misprediction cost appear
(7 → 18 cycles per element at 256K). So a benchmark either consumes ≥ 2^16 varied inputs
per sample, or reshuffles them in an untimed `reset`. Inputs come from a SplitMix64
generator in the driver, so they are identical on every Julia version, and each
benchmark seeds its own, so adding a benchmark never changes another's inputs.

**Calibrated by an A/A run.** Both arms the same build, the whole suite, three blocks:
no benchmark and no category was flagged. Single-threaded benchmarks have a median
worst-block deviation of 1.25% in cycles (90th percentile 5%). The tail is
allocation-heavy kernels, which carry wider tolerances.

**And by pairs of builds that should not differ.** An A/A run cannot see code placement:
two builds lay out the sysimage and the runtime differently even when the commit between
them touches only tests, lowering or JuliaSyntax (the build path is compiled in, so not
even the same commit built by two pipelines is bit-identical). Every block runs the same
two binaries, so blocks do not average this out. Over five consecutive julia-ci master
pairs of that kind, kernels whose hot code is in the sysimage (`Dict{Int,Int}`, integer
`sort!`, `exp`, `sincos`, `log`) moved 4-12% in cycles with identical instruction counts,
and each pair flagged two to four of them. So a cycles change that the instruction counts
do not explain (the same direction, at least half the size) must exceed 15%, which
cleared all five pairs; a change with an instruction change behind it keeps the
benchmark's own tolerance. Runtime dispatch goes through a call cache keyed on the
caller's address, so the dynamic-dispatch kernels move a few percent in instructions as
well and use 6%. A comparison of two builds takes about 6.5 minutes on the M4 Pro.

**A bespoke driver.** BenchmarkTools and Chairmarks were measured against the driver
here on the same kernels. With equal time budgets their estimates are no more stable
(above 100 µs per sample, all three are within 1.8–5% CV across processes), and
BenchmarkTools' default spends 10 s per benchmark to get a lower minimum, not a more
stable one. What neither offers is a hook around each sample for counters, an untimed
per-sample reset with a fixed evaluation count, and a record format the comparison needs.
`driver/Driver.jl` is ~150 lines.

## Categories

| Category | Covers | Judged on |
|---|---|---|
| `scalar` | Julia's libm (exp, log, sincos, pow), vectorizable Float32 map, integer div/rem/fld/mod, Int128, Float16, checked arithmetic and conversions | cycles |
| `simd` | reductions with and without `@simd`, Base's mapreduce, NaN-aware max/min, FMA dot, aliasing runtime checks, fused and 2-D broadcast, selects, narrow integers, gathers, tuple SLP, complex | cycles |
| `array` | Cartesian stencil with bounds checks, loop matmul, strided views, reinterpret, permutedims, push!/Memory growth, sort (float, int radix, by=), searchsorted, BitArray, cat/splat, stores and copies of a `Union{Missing,Float64}` array | cycles |
| `collections` | Dict with int, string and `Any` keys, deletion churn, IdDict, Set operations, iterator composition, tuples longer than 32 | cycles |
| `strings` | split/join, interpolation, number parsing and printing, regex captures, Unicode iteration, replace/search, symbol interning | cycles |
| `io` | IOBuffer print and readline, Serialization, a page-cache file round trip | cycles |
| `dispatch` | abstract element types, union splitting, isbits union returns, `Any` arithmetic, boxed closures, dynamic kwcalls, dynamic splats, invokelatest | cycles |
| `types` | subtyping, intersection, apply_type with the type cache, method lookup, type printing | cycles |
| `runtime` | throw/catch, try without throw, egal and hash, deepcopy, global bindings, task-local storage, a `ccall` with a `Ref` out-parameter, atomic fields, ScopedValues | cycles |
| `gc` | binary trees, full GC of a large heap, the write barrier's slow path over many old owners, big arrays, small arrays, finalizers, BigInt | time |
| `compiler` | parsing, macro expansion, lowering (whole file, closures and keywords, a large function), inference of a workload and of Base for a new type, first call, LLVM on a large function, method invalidation | cycles |
| `apps` | n-body, Mandelbrot, spectral norm, fannkuch, a JSON parser, a bytecode interpreter, LU with bounds checks, RK4 on tuples, BFS | cycles |
| `parallel` (4 threads) | spawn/fetch, nested spawn, static `@threads`, rebalancing an imbalanced loop (`:greedy`), channels (ping-pong, buffered), lock and atomic contention, task switches, threaded allocation, parallel marking of a broad forest | time |

### What a benchmark must satisfy

- It covers a code path no other benchmark covers. The question to ask is which code
  path it adds, not which parameter value.
- At least ~20 µs per call, and at most ~20 ms where possible, so it gets enough samples
  within its budget (0.5 s by default).
- Branchy code consumes ≥ 2^16 varied inputs, or reshuffles them in `reset`.
- Allocations it means to measure escape (stored into a container from `setup`). The
  compiler removes non-escaping allocations, and inlines the finalizers of non-escaping
  objects, and both have silently emptied benchmarks here.
- It returns what it constructs (the concatenated arrays, the copy, the deserialized
  value), not a length or a count: the driver keeps the returned value alive, but a
  length does not keep alive the work that built the contents.
- Inputs come from `SplitMix(seed)` inside its own `setup`, never from `rand`.
- Kernels that stream several arrays take them from `carve(T, n, k)`: one allocation at
  fixed offsets. Separately allocated arrays land at different relative alignments in
  every process. That moved `simd/axpy_alias_check` by 27% in cycles between processes
  with identical instructions; carved, the spread is 0.8%.
- It runs on Julia 1.10. A newer-only feature goes behind a check (see
  `runtime/scoped_values`).
- Its name stays the same. History on perf.julialang.org is keyed by name.

### Deliberately not covered

- **Libraries:** BLAS/LAPACK, SparseArrays, Statistics, Dates. These change with their
  own releases, and BaseBenchmarks keeps covering them.
- **Disk and network I/O:** too noisy on CI machines. The single file round trip stays in
  the page cache.
- **Package load time and precompilation:** that is the TTFX suite.
- **Parameter sweeps of the same kernel:** sizes, element types, `exp` on a hundred
  arguments.

## Output

`bench.jl` writes `results.json`, an array of records, one per benchmark, arm and block:

| field | |
|---|---|
| `name`, `category`, `arm`, `block`, `order` | identity; `order` is the global run order |
| `status`, `error` | `ok`, `error` or `skipped` |
| `metric`, `tolerance` | what the comparison judges (`cycles` or `time`), and a per-benchmark tolerance override |
| `times_ns`, `min_ns`, `median_ns` | every sample's wall time |
| `counters` | `{cycles: [...], instructions: [...], ...}` per sample; `counters_backend` says which backend recorded them |
| `allocs`, `bytes`, `gc_ns` | minimum allocations and bytes per sample, and GC time per sample |
| `ops` | work items per call, for time per item |
| `julia_version`, `threads` | as seen by the benchmark process |

It also writes `results-meta.json`: the builds (version, commit, LLVM, CPU target), the
machine, the settings, the suite commit, Buildkite variables and any failed processes.
`compare.jl` adds `report.md` and, for two arms, `compare.json` with the per-benchmark and
per-category verdicts. It exits with 1 on a regression, 3 on improvements only, and 2 when
a benchmark raised an error or a process failed (with one arm too); the report lists such
benchmarks, and any with missing blocks, under "Incomplete". A benchmark whose records
mix cycles and time (counters missing on one arm) is judged on time.

## Defining a benchmark

A benchmark is an ordinary function, registered with the driver in a separate line:

```julia
# benchmarks/simd.jl, inside the category's module
function axpy_alias_check_setup()
    y, x = carve(Float64, 2^18, 2)
    copyto!(y, rand_f64(SplitMix(6), 2^18))
    copyto!(x, rand_f64(SplitMix(7), 2^18))
    return y, x
end

# No @simd: two arrays that may alias, so LLVM must emit a runtime overlap check
function axpy_alias_check(y, x)
    @inbounds for i in eachindex(x, y)
        y[i] += 1.0000001 * x[i]
    end
    return y
end

bench(axpy_alias_check, axpy_alias_check_setup; ops = 2^18)
```

The benchmark is named after the kernel (`simd/axpy_alias_check`). `setup()` returns the
kernel's arguments, built once, and one sample is one call `kernel(args...)`. Keywords:
`reset` (called with the same arguments, untimed, before each sample, for kernels that
mutate their input), `ops` (work items per call), `seconds` (sampling budget), `metric`
(`"cycles"` or `"time"`), `tolerance` (overrides the comparison threshold), `min_threads`.

Each benchmark is a self-contained block, its setup and helpers directly above its
`bench` line, with literal sizes rather than file-level constants. A few files
(`compiler`, `types`, `runtime`, `strings`) start with a part shared by several of their
benchmarks, between `# Shared by the benchmarks below` and `# End of the shared part.`.
`julia -t 4 tools/check_blocks.jl` checks that every block runs on its own.

## Running one benchmark locally

Paste the functions into a REPL and call the kernel like any function, or time it with
whatever you like:

```julia
julia> axpy_alias_check(rand(1000), rand(1000))       # plain call

julia> include("driver/Driver.jl"); using .Driver    # the suite's harness

julia> bench(axpy_alias_check, axpy_alias_check_setup; ops = 2^18)
axpy_alias_check: min 57.46 µs (median 57.71 µs) · 0.2 ns/op · 226.1k cycles · 526.1k instructions · 0 allocs · 300 samples
```

Outside a suite run, `bench` runs the benchmark right away and returns the record (every
sample's time and counters). For the shared files, paste the shared part first;
`parallel` needs a REPL started with `-t 4`. From the shell,
`julia run.jl --category=simd --filter=axpy` runs it exactly as the suite does.
