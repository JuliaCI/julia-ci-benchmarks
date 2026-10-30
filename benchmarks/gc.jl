# The garbage collector, single-threaded: young-generation churn, marking a large old
# heap, write barriers, big objects, finalizers and malloc'd memory (GMP). Collections
# can run on GC threads, so these are judged on time with a wider tolerance. Every
# allocation escapes; otherwise the compiler removes it.
module GCBench
using ..Driver

# Short-lived trees: allocation fast path and young collections
struct TreeNode
    l::Union{TreeNode,Nothing}
    r::Union{TreeNode,Nothing}
end
make(d) = d == 0 ? TreeNode(nothing, nothing) : TreeNode(make(d - 1), make(d - 1))
check(t::TreeNode) = 1 + (t.l === nothing ? 0 : check(t.l)) + (t.r === nothing ? 0 : check(t.r))
# The trees escape into a ring from setup, each replacing one built a call earlier
binary_trees_setup() = Vector{Any}(nothing, 32)
function binary_trees(keep)
    s = 0
    for i in 1:32
        t = make(12)
        @inbounds keep[i] = t
        s += check(t)
    end
    return s
end
bench(binary_trees, binary_trees_setup; ops = 32 * 2^13, metric = "time", tolerance = 0.08)

# A full collection with ~1.5 million live objects in the old generation: mark speed
mutable struct ListNode
    val::Int
    next::Union{ListNode,Nothing}
    payload::Vector{Int}
end
function full_gc_large_heap_setup()
    head = nothing
    for i in 1:500_000
        head = ListNode(i, head, Int[i])
    end
    GC.gc(true)
    return Ref{Any}(head)
end
function full_gc_large_heap(h)
    GC.gc(true)
    return h
end
bench(full_gc_large_heap, full_gc_large_heap_setup; metric = "time", tolerance = 0.08, seconds = 1.0)

# Young objects stored into many old owners: the write barrier's slow path, which queues
# each owner in the remembered set. Once remembered, an owner takes the fast path until a
# full collection, so reset runs one before every sample.
function write_barrier_setup()
    owners = [Ref{Any}(nothing) for _ in 1:2^14]
    GC.gc(true)
    return owners
end
function write_barrier(owners)
    for i in eachindex(owners)
        @inbounds owners[i][] = Ref(i)
    end
    return owners
end
bench(write_barrier, write_barrier_setup; ops = 2^14, metric = "time", tolerance = 0.08,
      reset = owners -> GC.gc(true), seconds = 1.0)

# 1 MiB arrays allocated and dropped: the big-object path and malloc'd-memory accounting
large_arrays_setup() = Vector{Any}(nothing, 8)
function large_arrays(keep)
    s = 0.0
    for i in 1:256
        a = Vector{Float64}(undef, 2^17)
        a[1] = i
        a[end] = i
        keep[1 + i % 8] = a
        s += a[1] + a[end]
    end
    return s
end
bench(large_arrays, large_arrays_setup; ops = 256, metric = "time", tolerance = 0.2)

# Many small arrays: the pool allocator with array and Memory headers
small_arrays_setup() = Vector{Vector{Float64}}(undef, 256)
function small_arrays(keep)
    s = 0.0
    for i in 1:2^16
        a = [i, i + 1.0, i + 2.0]
        @inbounds keep[1 + i % 256] = a
        s += a[2]
    end
    return s
end
bench(small_arrays, small_arrays_setup; ops = 2^16, metric = "time", tolerance = 0.08)

# Objects with finalizers: registration, collection and running them. Escaping, since
# the finalizer of an object that does not escape is inlined.
mutable struct Handle
    id::Int
end
finalizers_setup() = Vector{Handle}(undef, 64)
const FINALIZED = Ref(0)
function finalizers(keep)
    for i in 1:2^15
        @inbounds keep[1 + i % 64] = finalizer(h -> (FINALIZED[] += 1; nothing), Handle(i))
    end
    GC.gc(false)
    return FINALIZED[]
end
bench(finalizers, finalizers_setup; ops = 2^15, metric = "time", tolerance = 0.1)

# BigInt arithmetic: GMP allocations through Julia's counted malloc, freed by finalizers
function bigint()
    a = big(1)
    b = big(1)
    for _ in 1:400
        a, b = b, a + b
        b = b * 3 ÷ 2 + 1
    end
    return b
end
bench(bigint; ops = 400, metric = "time", tolerance = 0.08)

end
