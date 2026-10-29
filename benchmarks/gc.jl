# The garbage collector, single-threaded: young-generation churn, marking a large old
# heap, write barriers, big objects, finalizers and malloc'd memory (GMP). Collections
# can run on GC threads, so these are judged on time with a wider tolerance. Every
# allocation escapes; otherwise the compiler removes it.
module GCBench
using ..Driver

struct TreeNode
    l::Union{TreeNode,Nothing}
    r::Union{TreeNode,Nothing}
end
make(d) = d == 0 ? TreeNode(nothing, nothing) : TreeNode(make(d - 1), make(d - 1))
check(t::TreeNode) = 1 + (t.l === nothing ? 0 : check(t.l)) + (t.r === nothing ? 0 : check(t.r))

# Short-lived trees: allocation fast path and young collections
bench("binary_trees"; ops = 32 * 2^13, metric = "time", tolerance = 0.08) do _
    s = 0
    for _ in 1:32
        s += check(make(12))
    end
    s
end

# A full collection with ~1.5 million live objects in the old generation: mark speed
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
bench("full_gc_large_heap"; setup = big_heap, metric = "time", tolerance = 0.08, seconds = 1.0) do h
    GC.gc(true)
    h
end

# Young objects stored into an old container: write barrier and remembered set
function old_container()
    v = Vector{Any}(undef, 2^16)
    fill!(v, nothing)
    GC.gc(true)
    return v
end
bench("write_barrier"; ops = 2^18, setup = old_container, metric = "time", tolerance = 0.08) do v
    n = length(v)
    for i in 1:2^18
        @inbounds v[1 + (i * 7919) % n] = Ref(i)
    end
    v
end

# 1 MiB arrays allocated and dropped: the big-object path and malloc'd-memory accounting
bench("large_arrays"; ops = 256, setup = () -> Vector{Any}(nothing, 8), metric = "time", tolerance = 0.2) do keep
    s = 0.0
    for i in 1:256
        a = Vector{Float64}(undef, 2^17)
        a[1] = i
        a[end] = i
        keep[1 + i % 8] = a
        s += a[1] + a[end]
    end
    s
end

# Many small arrays: the pool allocator with array and Memory headers
bench("small_arrays"; ops = 2^16, setup = () -> Vector{Vector{Float64}}(undef, 256), metric = "time", tolerance = 0.08) do keep
    s = 0.0
    for i in 1:2^16
        a = [i, i + 1.0, i + 2.0]
        @inbounds keep[1 + i % 256] = a
        s += a[2]
    end
    s
end

mutable struct Handle
    id::Int
end
const FINALIZED = Ref(0)
# Objects with finalizers: registration, collection and running them. Escaping, since
# the finalizer of an object that does not escape is inlined.
bench("finalizers"; ops = 2^15, setup = () -> Vector{Handle}(undef, 64), metric = "time", tolerance = 0.1) do keep
    for i in 1:2^15
        @inbounds keep[1 + i % 64] = finalizer(h -> (FINALIZED[] += 1; nothing), Handle(i))
    end
    GC.gc(false)
    FINALIZED[]
end

# BigInt arithmetic: GMP allocations through Julia's counted malloc, freed by finalizers
bench("bigint"; ops = 400, metric = "time", tolerance = 0.08) do _
    a = big(1)
    b = big(1)
    for _ in 1:400
        a, b = b, a + b
        b = b * 3 ÷ 2 + 1
    end
    ndigits(b)
end

end
