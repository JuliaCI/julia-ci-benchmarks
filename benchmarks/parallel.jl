# The scheduler, synchronization and multi-threaded GC. bench.jl runs this category with
# several threads (4 by default). Work spans threads, so all of these are judged on time,
# with tolerances that reflect the extra noise of cross-core communication.
module ParallelBench
using ..Driver
using Base.Threads

const NT = 4
opts = (; min_threads = NT, metric = "time", tolerance = 0.15)

# Many trivial tasks: creation, enqueue, wakeup, fetch
bench("spawn_fetch"; ops = 4096, opts...) do _
    ts = [Threads.@spawn(i + 1) for i in 1:4096]
    s = 0
    for t in ts
        s += fetch(t)
    end
    s
end

# Nested fork-join: recursive spawning with a sequential cutoff (work distribution)
fib(n) = n < 2 ? n : fib(n - 1) + fib(n - 2)
function pfib(n)
    n < 16 && return fib(n)
    t = Threads.@spawn pfib(n - 1)
    return pfib(n - 2) + fetch(t)::Int
end
bench("spawn_tree"; ops = 1, opts...) do _
    pfib(28)
end

# Fork-join loop overhead and memory bandwidth, static schedule
bench("threads_static_sum"; ops = 2^22, setup = () -> (rand_f64(SplitMix(1), 2^22), zeros(NT * 8)), opts..., tolerance = 0.2) do (x, part)
    fill!(part, 0.0)
    n = length(x)
    @threads :static for t in 1:NT
        lo = 1 + (t - 1) * n ÷ NT
        hi = t * n ÷ NT
        s = 0.0
        @inbounds @simd for i in lo:hi
            s += x[i]
        end
        part[8t] = s  # one cache line apart
    end
    sum(part)
end

# Imbalanced iterations under the dynamic schedule
work(i) = (s = 0.0; for k in 1:(50 * (1 + i % 17)); s += sqrt(k + i); end; s)
bench("threads_dynamic_imbalanced"; ops = 4096, setup = () -> zeros(4096), opts...) do out
    @threads :dynamic for i in 1:4096
        @inbounds out[i] = work(i)
    end
    out
end

# Unbuffered channel: every put! waits for a take! on another thread
bench("channel_pingpong"; ops = 512, opts...) do _
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

# Buffered producer/consumer throughput, two producers and two consumers
bench("channel_buffered"; ops = 2^16, opts...) do _
    ch = Channel{Int}(256)
    total = Threads.Atomic{Int}(0)
    producers = [Threads.@spawn(for i in 1:2^15; put!(ch, i); end) for _ in 1:2]
    consumers = [Threads.@spawn(for v in ch; Threads.atomic_add!(total, v); end) for _ in 1:2]
    foreach(wait, producers)
    close(ch)
    foreach(wait, consumers)
    total[]
end

# A contended ReentrantLock
bench("lock_contention"; ops = NT * 2^13, setup = () -> (ReentrantLock(), Ref(0)), opts...) do (lk, cnt)
    @sync for _ in 1:NT
        Threads.@spawn for _ in 1:2^13
            lock(lk) do
                cnt[] += 1
            end
        end
    end
    cnt[]
end

# Contended atomic read-modify-write: LSE atomics vs. LL/SC loops are a codegen choice
bench("atomic_contention"; ops = NT * 2^16, opts...) do _
    a = Threads.Atomic{Int}(0)
    @threads :static for _ in 1:NT
        for _ in 1:2^16
            Threads.atomic_add!(a, 1)
        end
    end
    a[]
end

# Task switches on one thread: two @async tasks handing control back and forth
bench("task_switch"; ops = 1024, opts...) do _
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

# Allocation on every thread at once: per-thread pools and stop-the-world collections
struct Pt
    x::Float64
    y::Float64
    tag::Any
end
bench("alloc_threads"; ops = NT * 2^16, setup = () -> [Vector{Any}(undef, 512) for _ in 1:NT], opts...) do keeps
    @threads :static for t in 1:NT
        keep = keeps[t]
        for i in 1:2^16
            @inbounds keep[1 + i % 512] = Pt(i, t, i)
        end
    end
    keeps
end

# Parallel marking of a large heap
mutable struct ListNode
    val::Int
    next::Union{ListNode,Nothing}
    payload::Vector{Int}
end
function big_heap()
    head = nothing
    for i in 1:500_000
        head = ListNode(i, head, Int[i])
    end
    GC.gc(true)
    return Ref{Any}(head)
end
bench("full_gc_parallel_mark"; setup = big_heap, seconds = 1.0, opts...) do h
    GC.gc(true)
    h
end

end
