# Runtime services outside dispatch: exceptions and unwinding, egal and hashing,
# reflection-driven copying, global bindings, scoped and task-local state.
module RuntimeBench
using ..Driver

struct ParseFailure <: Exception
    i::Int
end
@noinline check(x, i) = x > 0.9 ? throw(ParseFailure(i)) : x

# Every tenth element throws: jl_throw, unwinding and the backtrace collection
bench("throw_catch"; ops = 2^11, setup = () -> rand_f64(SplitMix(1), 2^11)) do xs
    s = 0.0
    for (i, x) in enumerate(xs)
        s += try
            check(x, i)
        catch e
            e isa ParseFailure ? -1.0 : rethrow()
        end
    end
    s
end

# try blocks that never throw, around cheap work: the cost of the handler itself
bench("try_nothrow"; ops = 2^16, setup = () -> rand_f64(SplitMix(2), 2^16) .* 0.8) do xs
    s = 0.0
    for (i, x) in enumerate(xs)
        s += try
            check(x, i)
        catch
            0.0
        end
    end
    s
end

# jl_egal and hashing of nested immutable values containing references
struct Key
    name::String
    tag::Symbol
    pos::Tuple{Int,Int}
    w::Float64
end
function keys_input()
    r = SplitMix(3)
    ks = [Key("k$(i % 500)", Symbol("t", i % 7), (i % 13, i % 17), (i % 5) / 4) for i in rand_int(r, 2^14, 1, 10^6)]
    return ks, circshift(ks, 1)
end
bench("egal_hash"; ops = 2^14, setup = keys_input) do (a, b)
    n = UInt(0)
    for i in eachindex(a)
        n += (a[i] === b[i]) + isequal(a[i], b[i]) + hash(a[i]) % 4
    end
    n
end

# deepcopy walks objects by reflection and tracks identity in an IdDict
mutable struct Node
    val::Int
    children::Vector{Node}
    meta::Dict{Symbol,Any}
end
function tree_input()
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
bench("deepcopy"; ops = 2001, setup = tree_input) do root
    length(deepcopy(root).children)
end

# Untyped and typed globals: binding lookup and world-age partitions on every access
global untyped_counter = 0
global typed_counter::Int = 0
bench("global_access"; ops = 2^16) do _
    global untyped_counter, typed_counter
    for i in 1:2^15
        untyped_counter += i
        typed_counter += i
    end
    untyped_counter + typed_counter
end

bench("task_local_storage"; ops = 2^14) do _
    s = 0
    for i in 1:2^14
        task_local_storage(:cibench_k, i)
        s += task_local_storage(:cibench_k)::Int
    end
    s
end

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
    bench("scoped_values"; ops = 2^12) do _
        s = 0.0
        for k in 1:256
            s += Base.ScopedValues.with(() -> sv_work(16), SV => k, SV2 => 0.5)
        end
        s
    end
end

end
