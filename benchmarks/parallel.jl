# The scheduler, synchronization and multi-threaded GC, on 4 threads: bench.jl starts this
# category with -t 4 (start the REPL with -t 4 to run a block locally). Work spans threads,
# so all of these are judged on time, with tolerances for the noise of cross-core traffic.
module ParallelBench
using ..Driver

# Many trivial tasks: creation, enqueue, wakeup, fetch
function spawn_fetch()
    ts = [Threads.@spawn(i + 1) for i in 1:4096]
    s = 0
    for t in ts
        s += fetch(t)
    end
    s
end
bench(spawn_fetch; ops = 4096, min_threads = 4, metric = "time", tolerance = 0.15)

# Nested fork-join: recursive spawning with a sequential cutoff (work distribution)
fib(n) = n < 2 ? n : fib(n - 1) + fib(n - 2)
function pfib(n)
    n < 16 && return fib(n)
    t = Threads.@spawn pfib(n - 1)
    return pfib(n - 2) + fetch(t)::Int
end
spawn_tree() = pfib(28)
bench(spawn_tree; ops = 1, min_threads = 4, metric = "time", tolerance = 0.15)

# Fork-join loop overhead and memory bandwidth, static schedule
threads_static_sum_setup() = (rand_f64(SplitMix(1), 2^22), zeros(4 * 8))
function threads_static_sum(x, part)
    fill!(part, 0.0)
    n = length(x)
    Threads.@threads :static for t in 1:4
        lo = 1 + (t - 1) * n ÷ 4
        hi = t * n ÷ 4
        s = 0.0
        @inbounds @simd for i in lo:hi
            s += x[i]
        end
        part[8t] = s  # one cache line apart
    end
    sum(part)
end
bench(threads_static_sum, threads_static_sum_setup; ops = 2^22, min_threads = 4, metric = "time", tolerance = 0.2)

# Imbalanced iterations: the first eighth costs 30x the rest, so equal contiguous chunks
# leave one thread with most of the work. `:dynamic` still splits the range into one chunk
# per thread; only `:greedy` (1.11+) hands out iterations as threads free up.
threads_dynamic_imbalanced_setup() = zeros(128)
work(i) = (s = 0.0; for k in 1:(i <= 16 ? 300_000 : 10_000); s += sqrt(k + i); end; s)
function threads_dynamic_imbalanced(out)
    @static if VERSION >= v"1.11"
        Threads.@threads :greedy for i in 1:128
            @inbounds out[i] = work(i)
        end
    else
        Threads.@threads :dynamic for i in 1:128
            @inbounds out[i] = work(i)
        end
    end
    out
end
bench(threads_dynamic_imbalanced, threads_dynamic_imbalanced_setup; ops = 128, min_threads = 4, metric = "time", tolerance = 0.15)

# Unbuffered channel: every put! waits for a take! on another thread
function channel_pingpong()
    a = Channel{Int}(0)
    b = Channel{Int}(0)
    t = Threads.@spawn for _ in 1:512
        put!(b, take!(a) + 1)
    end
    s = 0
    for i in 1:512
        put!(a, i)
        s += take!(b)
    end
    wait(t)
    s
end
bench(channel_pingpong; ops = 512, min_threads = 4, metric = "time", tolerance = 0.15)

# Buffered producer/consumer throughput, two producers and two consumers
function channel_buffered()
    ch = Channel{Int}(256)
    total = Threads.Atomic{Int}(0)
    producers = [Threads.@spawn(for i in 1:2^15; put!(ch, i); end) for _ in 1:2]
    consumers = [Threads.@spawn(for v in ch; Threads.atomic_add!(total, v); end) for _ in 1:2]
    foreach(wait, producers)
    close(ch)
    foreach(wait, consumers)
    total[]
end
bench(channel_buffered; ops = 2^16, min_threads = 4, metric = "time", tolerance = 0.15)

# A contended ReentrantLock
lock_contention_setup() = (ReentrantLock(), Ref(0))
function lock_contention(lk, cnt)
    @sync for _ in 1:4
        Threads.@spawn for _ in 1:2^13
            lock(lk) do
                cnt[] += 1
            end
        end
    end
    cnt[]
end
bench(lock_contention, lock_contention_setup; ops = 4 * 2^13, min_threads = 4, metric = "time", tolerance = 0.15)

# Contended atomic read-modify-write: LSE atomics vs. LL/SC loops are a codegen choice
function atomic_contention()
    a = Threads.Atomic{Int}(0)
    Threads.@threads :static for _ in 1:4
        for _ in 1:2^16
            Threads.atomic_add!(a, 1)
        end
    end
    a[]
end
bench(atomic_contention; ops = 4 * 2^16, min_threads = 4, metric = "time", tolerance = 0.15)

# Task switches on one thread: two @async tasks handing control back and forth
function task_switch()
    c1 = Condition()
    c2 = Condition()
    n = Ref(0)
    t = @async for _ in 1:512
        wait(c1)
        n[] += 1
        notify(c2)
    end
    yield()
    for _ in 1:512
        notify(c1)
        wait(c2)
    end
    wait(t)
    n[]
end
bench(task_switch; ops = 1024, min_threads = 4, metric = "time", tolerance = 0.15)

# Allocation on every thread at once: per-thread pools and stop-the-world collections
alloc_threads_setup() = [Vector{Any}(undef, 512) for _ in 1:4]
struct Pt
    x::Float64
    y::Float64
    tag::Any
end
function alloc_threads(keeps)
    Threads.@threads :static for t in 1:4
        keep = keeps[t]
        for i in 1:2^16
            @inbounds keep[1 + i % 512] = Pt(i, t, i)
        end
    end
    keeps
end
bench(alloc_threads, alloc_threads_setup; ops = 4 * 2^16, min_threads = 4, metric = "time", tolerance = 0.15)

# Parallel marking of a large heap: 64 trees of 8191 nodes, ~1.5 million objects, broad
# enough that the GC threads can split the work between them
mutable struct ForestNode
    l::Union{ForestNode,Nothing}
    r::Union{ForestNode,Nothing}
    payload::Vector{Int}
end
grow(d) = ForestNode(d == 0 ? nothing : grow(d - 1), d == 0 ? nothing : grow(d - 1), Int[d])
function full_gc_parallel_mark_setup()
    roots = Any[grow(12) for _ in 1:64]
    GC.gc(true)
    return roots
end
function full_gc_parallel_mark(roots)
    GC.gc(true)
    roots
end
bench(full_gc_parallel_mark, full_gc_parallel_mark_setup; seconds = 1.0, min_threads = 4, metric = "time", tolerance = 0.15)

end
