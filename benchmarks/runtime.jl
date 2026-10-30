# Runtime services outside dispatch: exceptions and unwinding, egal and hashing,
# reflection-driven copying, global bindings, scoped and task-local state.
module RuntimeBench
using ..Driver
using Logging

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
    return ks, circshift(ks, 1), SplitMix(103)
end
# The pairs are reshuffled every sample, together
function egal_hash(a, b, _)
    n = UInt(0)
    for i in eachindex(a)
        n += (a[i] === b[i]) + isequal(a[i], b[i]) + hash(a[i]) % 4
    end
    return n
end
bench(egal_hash, egal_hash_setup; ops = 2^14, reset = (a, b, r) -> shuffle_det!(r, a, b))

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
    return root, nodes, SplitMix(104)
end
# Every node's children are reordered before each sample, so the walk differs
deepcopy_tree(root, _, _) = deepcopy(root)
bench(deepcopy_tree, deepcopy_tree_setup; ops = 2001,
      reset = (root, nodes, r) -> foreach(n -> shuffle_det!(r, n.children), nodes))

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
const_field_roots_setup() = (WithConst(ntuple(_ -> zeros(1), 64)), rand_int(SplitMix(43), 2^14, 1, 64), SplitMix(143))
# The field choices are reshuffled every sample
function const_field_roots(s, ks, _)
    for k in ks
        Base.Cartesian.@nexprs 64 i -> i == k && bump!(s.cols[i])
    end
    return s.cols[1][1]
end
bench(const_field_roots, const_field_roots_setup; ops = 2^14, reset = (s, ks, r) -> shuffle_det!(r, ks))

# A backtrace turned into frames, as error display, logging and Test do it: the debug-info
# lookup of every instruction pointer, inlined frames included
@noinline deep_throw(n) = n == 0 ? error("bottom") : (deep_throw(n - 1); n)
function stacktrace_lookup_setup()
    try
        deep_throw(40)
    catch
        return catch_backtrace()
    end
end
stacktrace_lookup(bt) = stacktrace(bt)
bench(stacktrace_lookup, stacktrace_lookup_setup)

# Log statements: @debug below the minimum level (the early check), @info through a
# SimpleLogger into an IOBuffer (message and key-value formatting), and @info under
# with_logger(NullLogger()) (#55761). The buffer is emptied before every sample.
function log_statements_setup()
    io = IOBuffer()
    return io, Logging.SimpleLogger(io, Logging.Info), rand_int(SplitMix(79), 2^10, 0, 10^6)
end
function log_statements(io, logger, xs)
    with_logger(logger) do
        for (i, x) in enumerate(xs)
            @debug "skipped" x
            i % 8 == 0 && @info "value" x i
        end
    end
    with_logger(NullLogger()) do
        for x in xs
            @info "dropped" x
        end
    end
    return io
end
bench(log_statements, log_statements_setup; ops = 3 * 2^10, reset = (io, logger, xs) -> take!(io))

# Locks taken and released without contention, as every IO operation and many caches do:
# ReentrantLock, reentrantly too, @lock, and a SpinLock
lock_uncontended_setup() = (ReentrantLock(), Threads.SpinLock(), Ref(0))
function lock_uncontended(rl, sl, c)
    for i in 1:2^14
        lock(rl)
        c[] += i
        unlock(rl)
        Base.@lock rl begin
            lock(rl)
            c[] -= 1
            unlock(rl)
        end
        lock(sl)
        c[] += 1
        unlock(sl)
    end
    return c[]
end
bench(lock_uncontended, lock_uncontended_setup; ops = 3 * 2^14)

# Globals looked up by name at run time, as getproperty on a Module does: isdefined and
# getglobal on Base, through the module's binding table. A few names do not exist. The
# names are reshuffled every sample.
function module_bindings_setup()
    names = [:sum, :map, :filter, :push!, :length, :Dict, :Vector, :println, :string, :parse, :sort!, :copy,
             :getindex, :setindex!, :iterate, :zero, :one, :typemax, :Int, :Float64, :String, :Symbol, :pi,
             :stdout, :ENV, :VERSION, :nothing, :missing, :isless, :hash, :show, :print, :reduce, :foldl,
             :mapreduce, :findfirst, :replace, :split, :join, :abs, :sqrt, :exp, :log, :sin, :max, :min,
             :reverse, :unique, :keys, :values, :haskey, :get, :get!, :merge, :empty!, :resize!, :fill,
             :zeros, :no_such_name_1, :no_such_name_2, :no_such_name_3, :no_such_name_4, :Base, :Core]
    return repeat(names, 64), SplitMix(180)
end
function module_bindings(names, _)
    n = 0
    for s in names
        if isdefined(Base, s)
            n += getglobal(Base, s) isa Function
        end
    end
    return n
end
bench(module_bindings, module_bindings_setup; ops = 64 * 64, reset = (names, r) -> shuffle_det!(r, names))

# Strings passed to C and back: the Cstring conversion and its check for NUL bytes,
# strlen and strchr, and unsafe_string copying the result
function ccall_string_setup()
    r = SplitMix(80)
    strs = [String(UInt8.(rand_int(r, n, 0x61, 0x7a))) for n in rand_int(r, 4096, 1, 200)]
    return strs, Vector{String}(undef, 4096)
end
function ccall_string(strs, out)
    n = 0
    for (i, s) in enumerate(strs)
        n += Int(ccall(:strlen, Csize_t, (Cstring,), s))
        p = ccall(:strchr, Ptr{UInt8}, (Cstring, Cint), s, Cint('q'))
        out[i] = p == C_NULL ? "" : unsafe_string(p)
    end
    return n, out
end
bench(ccall_string, ccall_string_setup; ops = 4096)

# Large immutables passed to and returned from calls that are not inlined: twelve floats
# returned through a hidden pointer (sret), and a struct with references among its
# fields, whose GC roots are returned alongside
struct Big12
    t::NTuple{12,Float64}
end
struct WithRefs
    name::String
    a::Float64
    b::Float64
    v::Vector{Float64}
end
@noinline step12(s::Big12, x) = Big12(ntuple(i -> s.t[i] * 0.5 + x * i, Val(12)))
@noinline retag(w::WithRefs, x, names) = WithRefs(names[1 + (x > 0.5)], w.a + x, w.b * x, w.v)
struct_abi_setup() = (rand_f64(SplitMix(81), 2^14), ["low", "high"], [1.0])
function struct_abi(xs, names, v)
    s = Big12(ntuple(Float64, Val(12)))
    w = WithRefs("", 0.0, 1.0, v)
    for x in xs
        s = step12(s, x)
        w = retag(w, x, names)
    end
    return s, w
end
bench(struct_abi, struct_abi_setup; ops = 2 * 2^14)

# Mutable objects that never escape: allocation elimination must turn a Ref and a mutable
# struct into registers, inside the loop and carried around it. Losing it makes every
# iteration allocate.
mutable struct Acc
    s::Float64
    n::Int
end
@inline add!(a::Acc, x) = (a.s += x; a.n += 1; a)
nonescaping_mutable_setup() = rand_f64(SplitMix(82), 2^16)
function nonescaping_mutable(xs)
    tot = 0.0
    outer = Acc(0.0, 0)
    for x in xs
        a = Acc(0.0, 0)
        r = Ref(x)
        add!(a, r[])
        add!(a, 2 * r[])
        r[] = a.s / a.n
        tot += r[]
        add!(outer, x)
    end
    return tot + outer.s / outer.n
end
bench(nonescaping_mutable, nonescaping_mutable_setup; ops = 2^16)

# Recursion: the call overhead of a specialized signature (fib), and a recursive walk
# whose frames hold a reference across each call, so every frame roots it
fibr(n) = n < 2 ? n : fibr(n - 1) + fibr(n - 2)
struct TNode
    v::Vector{Int}
    kids::Vector{TNode}
end
function walk(t::TNode, d)
    s = t.v[1 + d % length(t.v)]
    for k in t.kids
        s += walk(k, d + 1)
    end
    return s + length(t.v)
end
function recursive_calls_setup()
    r = SplitMix(83)
    leaf() = TNode(rand_int(r, rand_int(r, 1, 1, 4)[1], 0, 100), TNode[])
    # Each node attaches to a random earlier one
    nodes = [leaf()]
    for _ in 1:2^13
        c = leaf()
        push!(nodes[rand_int(r, 1, 1, length(nodes))[1]].kids, c)
        push!(nodes, c)
    end
    return 21, nodes[1]
end
recursive_calls(n, t) = fibr(n) + walk(t, 0)
bench(recursive_calls, recursive_calls_setup)

end
