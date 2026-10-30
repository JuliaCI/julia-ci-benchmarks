# Hash tables and iterator composition: hashing, probing and rehash growth for concrete,
# string and untyped keys, the C-runtime IdDict, and long tuples past the inlining limits.
module CollectionsBench
using ..Driver

# Built from empty every sample, so rehash growth is included; half the lookups miss
function dict_int_setup()
    r = SplitMix(1)
    return rand_int(r, 2^16, 1, 2^18), rand_int(r, 2^16, 1, 2^18)
end
function dict_int(ks, qs)
    d = Dict{Int,Int}()
    for (i, k) in enumerate(ks)
        d[k] = i
    end
    s = 0
    for q in qs
        s += get(d, q, 0)
    end
    s
end
bench(dict_int, dict_int_setup; ops = 2^17)

# Word counting: string hashing and comparison. Every occurrence is its own String, so
# isequal compares contents rather than taking the identity fast path.
function dict_string_count_setup()
    r = SplitMix(2)
    vocab = [String([Char('a' + Int(Driver.next!(r) % 26)) for _ in 1:rand_int(r, 1, 3, 12)[1]]) for _ in 1:4096]
    return [String(Vector{UInt8}(vocab[i])) for i in rand_int(r, 2^16, 1, 4096)]
end
function dict_string_count(words)
    d = Dict{String,Int}()
    for w in words
        d[w] = get(d, w, 0) + 1
    end
    length(d)
end
bench(dict_string_count, dict_string_count_setup; ops = 2^16)

# Untyped keys and values, as in much of Base and package internals: every hash and
# isequal is a dynamic call
function dict_any_setup()
    r = SplitMix(3)
    pool = Any[1, 2.5, "x", :y, (1, 2), 'c', 0x05, 1 // 3, nothing, [1]]
    return Any[i % 3 == 0 ? string(i) : i % 3 == 1 ? i : Float64(i) for i in rand_int(r, 2^15, 1, 2^16)], pool
end
function dict_any(ks, pool)
    d = Dict{Any,Any}()
    for (i, k) in enumerate(ks)
        d[k] = pool[1 + i % length(pool)]
    end
    s = 0
    for k in ks
        s += d[k] isa Number
    end
    s
end
bench(dict_any, dict_any_setup; ops = 2^15)

# Deletion and reinsertion at a fixed population: tombstones, probing past deleted slots
# and the rehash that clears them. Reset rebuilds the Dict, so every sample starts alike.
function dict_delete_churn_setup()
    ks = shuffle_det!(SplitMix(6), collect(1:2^16))
    return Ref(Dict(k => k for k in ks)), ks
end
function dict_delete_churn(box, ks)
    d = box[]
    n = length(ks)
    s = 0
    for k in ks
        delete!(d, k)
        d[k + n] = k
        s += get(d, k, 0) + d[k + n]
    end
    return s
end
bench(dict_delete_churn, dict_delete_churn_setup; ops = 2^16, reset = (box, ks) -> (box[] = Dict(k => k for k in ks)))

# IdDict lives in the C runtime (eqtable), keyed by object identity
function iddict_setup()
    return [Ref(i) for i in 1:2^15]
end
function iddict(objs)
    d = IdDict{Any,Int}()
    for (i, o) in enumerate(objs)
        d[o] = i
    end
    s = 0
    for o in objs
        s += d[o]
    end
    s
end
bench(iddict, iddict_setup; ops = 2^15)

function set_ops_setup()
    r = SplitMix(4)
    return rand_int(r, 2^16, 1, 2^17), rand_int(r, 2^16, 1, 2^17)
end
function set_ops(a, b)
    sa, sb = Set(a), Set(b)
    length(union(sa, sb)) + length(intersect(sa, sb)) + length(setdiff(sa, sb)) + length(unique(a))
end
bench(set_ops, set_ops_setup; ops = 2^17)

# Generators, filters, zip, enumerate and flatten: all must inline to a plain loop
iterator_compose_setup() = rand_int(SplitMix(5), 2^18, -1000, 1000)
function iterator_compose(x)
    a = sum(v^2 for v in x if isodd(v))
    b = sum(i * v for (i, v) in enumerate(x) if v > 0)
    c = count(p -> p[1] < p[2], zip(x, Iterators.drop(x, 1)))
    d = sum(Iterators.flatten((x, x)))
    a + b + c + d
end
bench(iterator_compose, iterator_compose_setup; ops = 2^18)

# Tuples past the inference and inlining limits (length > 32): splats, maps and
# reductions over long heterogeneous and homogeneous tuples, new ones every iteration
long_tuples_setup() = (ntuple(i -> Float64(i), 40), ntuple(i -> isodd(i) ? i : Float64(i), 36))
function long_tuples(t, h)
    s = 0.0
    for k in 1:1000
        u = map(x -> x * k, t)
        v = map(x -> x * k, h)
        s += +(u...) + sum(v) + max(u...)
    end
    s
end
bench(long_tuples, long_tuples_setup; ops = 1000)


# findall with a predicate that folds over a 7-tuple (#56309: a cliff between N=6 and N=7)
has_one(t) = mapfoldl(==(1), |, t; init = false)
function findall_tuples_setup()
    v = rand_int(SplitMix(38), 7 * 2^16, 1, 64)
    return [ntuple(j -> v[7(i-1)+j], 7) for i in 1:2^16]
end
findall_tuples(v) = length(findall(has_one, v))
bench(findall_tuples, findall_tuples_setup; ops = 2^16)

# Reductions with init=, which take the mapfoldl path, including over zip with a view
# (#47216, #48792, #49763)
reduce_init_setup() = (v = rand_f64(SplitMix(39), 2^17); (v, view(v, :)))
reduce_init(v, w) = reduce(max, v; init = 0.0) + sum(x -> 4 / (4x - 3), v; init = 0.0) +
                    mapfoldl(t -> t[1] * t[2], +, zip(v, w); init = 0.0)
bench(reduce_init, reduce_init_setup; ops = 3 * 2^17)

# collect and map whose element type is only known at runtime: the eltype widens partway
# through (setindex_widen_up_to)
function collect_widen_setup()
    r = SplitMix(40)
    return Any[k == 1 ? 1.5 : k for k in rand_int(r, 2^15, 1, 4)]
end
function collect_widen(xs)
    a = [x + 1 for x in xs]
    b = map(x -> x isa Int ? nothing : x, xs)
    return length(a) + length(b) + count(isnothing, b)
end
bench(collect_widen, collect_widen_setup; ops = 2 * 2^15, tolerance = 0.06)

end
