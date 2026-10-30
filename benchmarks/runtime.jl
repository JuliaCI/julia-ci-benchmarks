# Runtime services outside dispatch: exceptions and unwinding, egal and hashing,
# reflection-driven copying, global bindings, scoped and task-local state.
module RuntimeBench
using ..Driver

# Shared by the benchmarks below: paste this part first to run one of them in the REPL.
struct ParseFailure <: Exception
    i::Int
end
@noinline check(x, i) = x > 0.9 ? throw(ParseFailure(i)) : x
# End of the shared part.

# Every tenth element throws: jl_throw, unwinding and the backtrace collection. The input
# is reshuffled before every sample, so which elements throw is unpredictable.
throw_catch_setup() = (rand_f64(SplitMix(1), 2^11), SplitMix(101))
function throw_catch(xs, r)
    s = 0.0
    for (i, x) in enumerate(xs)
        s += try
            check(x, i)
        catch e
            e isa ParseFailure ? -1.0 : rethrow()
        end
    end
    return s
end
bench(throw_catch, throw_catch_setup; ops = 2^11, reset = (xs, r) -> shuffle_det!(r, xs))

# try blocks that never throw, around cheap work: the cost of the handler itself
try_nothrow_setup() = rand_f64(SplitMix(2), 2^16) .* 0.8
function try_nothrow(xs)
    s = 0.0
    for (i, x) in enumerate(xs)
        s += try
            check(x, i)
        catch
            0.0
        end
    end
    return s
end
bench(try_nothrow, try_nothrow_setup; ops = 2^16)

# jl_egal and hashing of nested immutable values containing references
struct Key
    name::String
    tag::Symbol
    pos::Tuple{Int,Int}
    w::Float64
end
function egal_hash_setup()
    r = SplitMix(3)
    ks = [Key("k$(i % 500)", Symbol("t", i % 7), (i % 13, i % 17), (i % 5) / 4) for i in rand_int(r, 2^14, 1, 10^6)]
    return ks, circshift(ks, 1)
end
function egal_hash(a, b)
    n = UInt(0)
    for i in eachindex(a)
        n += (a[i] === b[i]) + isequal(a[i], b[i]) + hash(a[i]) % 4
    end
    return n
end
bench(egal_hash, egal_hash_setup; ops = 2^14)

# deepcopy walks objects by reflection and tracks identity in an IdDict
mutable struct Node
    val::Int
    children::Vector{Node}
    meta::Dict{Symbol,Any}
end
function deepcopy_tree_setup()
    root = Node(0, Node[], Dict{Symbol,Any}())
    nodes = [root]
    r = SplitMix(4)
    for i in 1:2000
        parent = nodes[rand_int(r, 1, 1, length(nodes))[1]]
        c = Node(i, Node[], Dict{Symbol,Any}(:i => i, :s => "n$i"))
        push!(parent.children, c)
        push!(nodes, c)
    end
    return root
end
deepcopy_tree(root) = deepcopy(root)
bench(deepcopy_tree, deepcopy_tree_setup; ops = 2001)

# Untyped and typed globals: binding lookup and world-age partitions on every access
global untyped_counter = 0
global typed_counter::Int = 0
function global_access()
    global untyped_counter, typed_counter
    for i in 1:2^15
        untyped_counter += i
        typed_counter += i
    end
    return untyped_counter + typed_counter
end
bench(global_access; ops = 2^16)

function task_local_storage_rw()
    s = 0
    for i in 1:2^14
        task_local_storage(:cibench_k, i)
        s += task_local_storage(:cibench_k)::Int
    end
    return s
end
bench(task_local_storage_rw; ops = 2^14)

# A cheap C call with a Ref out-parameter: ccall argument conversion, and the Ref kept on
# the stack rather than allocated
ccall_ref_output_setup() = rand_f64(SplitMix(5), 2^16, -1e6, 1e6)
function ccall_ref_output(xs)
    s = 0.0
    @inbounds for x in xs
        ipart = Ref{Cdouble}(0.0)
        fpart = ccall((:modf, Base.Math.libm), Cdouble, (Cdouble, Ref{Cdouble}), x, ipart)
        s += fpart + 0.125 * ipart[]
    end
    return s
end
bench(ccall_ref_output, ccall_ref_output_setup; ops = 2^16)

# Atomic fields, uncontended: atomic loads and compare-exchange that succeeds about half
# the time, with explicit memory orderings
mutable struct AtomicSlot
    @atomic value::Int
end
atomic_field_cas_setup() = (AtomicSlot(0), rand_int(SplitMix(6), 2^16, -10^6, 10^6))
function atomic_field_cas(slot, xs)
    successes = 0
    @inbounds for x in xs
        old = @atomic :acquire slot.value
        expected = isodd(x) ? old : xor(old, 1)
        result = @atomicreplace :acquire_release :acquire slot.value expected => x
        successes += result.success
    end
    return successes, (@atomic :acquire slot.value)
end
bench(atomic_field_cas, atomic_field_cas_setup; ops = 2^16, reset = (slot, xs) -> (@atomic :release slot.value = 0))

# ScopedValues (1.11+): HAMT lookups and dynamic scopes
if isdefined(Base, :ScopedValues)
    const SV = Base.ScopedValues.ScopedValue(0)
    const SV2 = Base.ScopedValues.ScopedValue(1.0)
    function sv_work(n)
        s = 0.0
        for i in 1:n
            s += SV[] * SV2[]
        end
        return s
    end
    function scoped_values()
        s = 0.0
        for k in 1:256
            s += Base.ScopedValues.with(() -> sv_work(16), SV => k, SV2 => 0.5)
        end
        return s
    end
    bench(scoped_values; ops = 2^12)
end


# Calls inside a loop over the const fields of a mutable struct: the loads are invariant,
# and losing that keeps every field live in the GC frame across each call (#63480)
mutable struct WithConst{S<:Tuple}
    const cols::S
end
@noinline bump!(c::Vector{Float64}) = (@inbounds c[1] += 1.0; nothing)
const_field_roots_setup() = (WithConst(ntuple(_ -> zeros(1), 64)), rand_int(SplitMix(43), 2^14, 1, 64))
function const_field_roots(s, ks)
    for k in ks
        Base.Cartesian.@nexprs 64 i -> i == k && bump!(s.cols[i])
    end
    return s.cols[1][1]
end
bench(const_field_roots, const_field_roots_setup; ops = 2^14)

end
