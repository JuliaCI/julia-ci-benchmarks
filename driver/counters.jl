# Hardware counters of the calling thread, read around every sample. Backends:
#
#   :kperf   macOS, root only: cycles, instructions, branches, branch misses, L1D load misses
#            through the private kperf/kperfdata frameworks
#   :thread  macOS, no privileges: cycles and instructions from `thread_selfcounts`
#   :perf    Linux: user-space cycles and instructions of the calling thread through
#            perf_event_open, which needs kernel.perf_event_paranoid <= 2 (the default
#            on most distributions) and a container that allows the syscall
#   :none    anything else
#
# `CIBENCH_COUNTERS` (auto, kperf, thread, perf, none) overrides the automatic choice.
struct CounterBackend
    kind::Symbol
    names::Vector{String}
    map::Vector{Int}          # event k is at buf[map[k]] in the backend's buffer
    buf::Vector{UInt64}
    fd::Cint                  # perf: the group leader's file descriptor
end
CounterBackend(kind, names, map, buf) = CounterBackend(kind, names, map, buf, Cint(-1))

const NO_COUNTERS = CounterBackend(:none, String[], Int[], UInt64[])

const KPERF = "/System/Library/PrivateFrameworks/kperf.framework/kperf"
const KPERFDATA = "/System/Library/PrivateFrameworks/kperfdata.framework/kperfdata"
const KPC_MAX_COUNTERS = 32
# Our name => the event names that mean it, newest CPUs first
const KPERF_EVENTS = [
    "cycles" => ["FIXED_CYCLES", "CPU_CLK_UNHALTED.THREAD", "CPU_CLK_UNHALTED.CORE"],
    "instructions" => ["FIXED_INSTRUCTIONS", "INST_RETIRED.ANY"],
    "branches" => ["INST_BRANCH", "BR_INST_RETIRED.ALL_BRANCHES"],
    "branch_misses" => ["BRANCH_MISPRED_NONSPEC", "BRANCH_MISPREDICT", "BR_MISP_RETIRED.ALL_BRANCHES"],
    "l1d_misses" => ["L1D_CACHE_MISS_LD_NONSPEC", "L1D_CACHE_MISS_LD", "MEM_LOAD_RETIRED.L1_MISS"],
]

function counter_backend()
    want = get(ENV, "CIBENCH_COUNTERS", "auto")
    want == "none" && return NO_COUNTERS
    if Sys.islinux()
        want in ("auto", "perf") || error("unknown CIBENCH_COUNTERS=$want on Linux")
        b = perf_backend()
        b === nothing && want == "perf" && @warn "perf_event_open counters unavailable"
        return something(b, NO_COUNTERS)
    end
    Sys.isapple() || return NO_COUNTERS
    if want in ("auto", "kperf")
        b = try
            kperf_backend()
        catch err
            want == "kperf" && @warn "kperf counters unavailable" exception = err
            nothing
        end
        b === nothing || return b
    end
    want in ("auto", "thread", "kperf") || error("unknown CIBENCH_COUNTERS=$want")
    buf = zeros(UInt64, 2)
    rc = ccall(:thread_selfcounts, Cint, (Cint, Ptr{UInt64}, Csize_t), 1, buf, sizeof(buf))
    rc == 0 || return NO_COUNTERS
    # thread_selfcounts(1) fills {instructions, cycles}
    return CounterBackend(:thread, ["instructions", "cycles"], [1, 2], buf)
end

function read_counters!(b::CounterBackend, out::Vector{UInt64})
    if b.kind === :thread
        ccall(:thread_selfcounts, Cint, (Cint, Ptr{UInt64}, Csize_t), 1, b.buf, sizeof(b.buf))
    elseif b.kind === :perf
        # One read of the group leader gives {number of events, cycles, instructions}
        ccall(:read, Cssize_t, (Cint, Ptr{UInt64}, Csize_t), b.fd, b.buf, sizeof(b.buf))
    elseif b.kind === :kperf
        ccall(KPERF_SYM[][:kpc_get_thread_counters], Cint, (UInt32, UInt32, Ptr{UInt64}),
              0, KPC_MAX_COUNTERS, b.buf)
    end
    for k in eachindex(b.names)
        out[k] = b.buf[b.map[k]]
    end
    return out
end

const KPERF_SYM = Ref{Dict{Symbol,Ptr{Cvoid}}}()

# Follows the kpc_demo by ibireme: resolve each event in the CPU's kpep database, let
# kperfdata compute the PMC configuration, then program the counters for this thread.
function kperf_backend()
    Sys.isapple() || return nothing
    hk = Libc.Libdl.dlopen(KPERF; throw_error = false)
    hd = Libc.Libdl.dlopen(KPERFDATA; throw_error = false)
    (hk === nothing || hd === nothing) && return nothing
    sym = Dict{Symbol,Ptr{Cvoid}}()
    for s in (:kpc_force_all_ctrs_set, :kpc_set_config, :kpc_set_counting,
              :kpc_set_thread_counting, :kpc_get_thread_counters)
        sym[s] = Libc.Libdl.dlsym(hk, s)
    end
    for s in (:kpep_db_create, :kpep_config_create, :kpep_config_force_counters,
              :kpep_db_event, :kpep_config_add_event, :kpep_config_kpc_classes,
              :kpep_config_kpc_count, :kpep_config_kpc_map, :kpep_config_kpc)
        sym[s] = Libc.Libdl.dlsym(hd, s)
    end
    # Needs root; fails with EPERM otherwise
    ccall(sym[:kpc_force_all_ctrs_set], Cint, (Cint,), 1) == 0 || return nothing
    db = Ref{Ptr{Cvoid}}(C_NULL)
    ccall(sym[:kpep_db_create], Cint, (Ptr{Cchar}, Ref{Ptr{Cvoid}}), C_NULL, db) == 0 ||
        error("kpep_db_create failed")
    cfg = Ref{Ptr{Cvoid}}(C_NULL)
    ccall(sym[:kpep_config_create], Cint, (Ptr{Cvoid}, Ref{Ptr{Cvoid}}), db[], cfg) == 0 ||
        error("kpep_config_create failed")
    ccall(sym[:kpep_config_force_counters], Cint, (Ptr{Cvoid},), cfg[]) == 0 ||
        error("kpep_config_force_counters failed")
    names = String[]
    for (name, aliases) in KPERF_EVENTS
        for alias in aliases
            ev = Ref{Ptr{Cvoid}}(C_NULL)
            ccall(sym[:kpep_db_event], Cint, (Ptr{Cvoid}, Cstring, Ref{Ptr{Cvoid}}), db[], alias, ev) == 0 || continue
            err = Ref{UInt32}(0)
            ccall(sym[:kpep_config_add_event], Cint, (Ptr{Cvoid}, Ref{Ptr{Cvoid}}, UInt32, Ref{UInt32}),
                  cfg[], ev, 0, err) == 0 || continue
            push!(names, name)
            break
        end
    end
    isempty(names) && error("no kperf events available")
    classes = Ref{UInt32}(0)
    ccall(sym[:kpep_config_kpc_classes], Cint, (Ptr{Cvoid}, Ref{UInt32}), cfg[], classes) == 0 ||
        error("kpep_config_kpc_classes failed")
    nregs = Ref{Csize_t}(0)
    ccall(sym[:kpep_config_kpc_count], Cint, (Ptr{Cvoid}, Ref{Csize_t}), cfg[], nregs) == 0 ||
        error("kpep_config_kpc_count failed")
    cmap = zeros(Csize_t, KPC_MAX_COUNTERS)
    ccall(sym[:kpep_config_kpc_map], Cint, (Ptr{Cvoid}, Ptr{Csize_t}, Csize_t), cfg[], cmap, sizeof(cmap)) == 0 ||
        error("kpep_config_kpc_map failed")
    regs = zeros(UInt64, KPC_MAX_COUNTERS)
    ccall(sym[:kpep_config_kpc], Cint, (Ptr{Cvoid}, Ptr{UInt64}, Csize_t), cfg[], regs, sizeof(regs)) == 0 ||
        error("kpep_config_kpc failed")
    ccall(sym[:kpc_set_config], Cint, (UInt32, Ptr{UInt64}), classes[], regs) == 0 ||
        error("kpc_set_config failed")
    ccall(sym[:kpc_set_counting], Cint, (UInt32,), classes[]) == 0 || error("kpc_set_counting failed")
    ccall(sym[:kpc_set_thread_counting], Cint, (UInt32,), classes[]) == 0 ||
        error("kpc_set_thread_counting failed")
    KPERF_SYM[] = sym
    return CounterBackend(:kperf, names, [Int(cmap[k]) + 1 for k in eachindex(names)],
                          zeros(UInt64, KPC_MAX_COUNTERS))
end

# Linux perf_event_open: a group of two hardware events, user space only, counting the
# calling thread (pid 0, any CPU) from the moment they are opened
const PERF_EVENT_OPEN = Dict(:x86_64 => 298, :aarch64 => 241, :i686 => 336, :powerpc64le => 319, :riscv64 => 241)
function perf_open(config::UInt64, group::Cint)
    attr = zeros(UInt8, 128)
    put!(T, off, v) = unsafe_store!(Ptr{T}(pointer(attr, off + 1)), v)
    fd = GC.@preserve attr begin
        put!(UInt32, 0, UInt32(0))                  # type: PERF_TYPE_HARDWARE
        put!(UInt32, 4, UInt32(128))                # size of the struct
        put!(UInt64, 8, config)                     # 0 cycles, 1 instructions
        put!(UInt64, 32, UInt64(0x8))               # read_format: PERF_FORMAT_GROUP
        put!(UInt64, 40, UInt64((1 << 5) | (1 << 6))) # exclude_kernel, exclude_hv
        ccall(:syscall, Clong, (Clong, Ptr{UInt8}, Cint, Cint, Cint, Culong),
              PERF_EVENT_OPEN[Sys.ARCH], attr, 0, -1, group, 0)
    end
    return Cint(fd)
end
function perf_backend()
    haskey(PERF_EVENT_OPEN, Sys.ARCH) || return nothing
    leader = perf_open(UInt64(0), Cint(-1))
    leader < 0 && return nothing
    if perf_open(UInt64(1), leader) < 0
        ccall(:close, Cint, (Cint,), leader)
        return nothing
    end
    b = CounterBackend(:perf, ["cycles", "instructions"], [2, 3], zeros(UInt64, 3), leader)
    # A virtual machine without a PMU opens the events but never counts
    read_counters!(b, zeros(UInt64, 2))
    x = b.buf[3]
    s = 0
    for i in 1:1000
        s += i
    end
    Base.donotdelete(s)
    read_counters!(b, zeros(UInt64, 2))
    return b.buf[1] == 2 && b.buf[3] > x ? b : nothing
end
