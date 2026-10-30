# Hash tables and iterator composition: hashing, probing and rehash growth for concrete,
# string and untyped keys, the C-runtime IdDict, and long tuples past the inlining limits.
module CollectionsBench
using ..Driver

# Built from empty every sample, so rehash growth is included; about a fifth of the
# lookups hit
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
    return d
end
bench(dict_string_count, dict_string_count_setup; ops = 2^16)

# Untyped keys and values, as in much of Base and package internals: every hash and
# isequal is a dynamic call. The keys are reshuffled every sample.
function dict_any_setup()
    r = SplitMix(3)
    pool = Any[1, 2.5, "x", :y, (1, 2), 'c', 0x05, 1 // 3, nothing, [1]]
    return Any[i % 3 == 0 ? string(i) : i % 3 == 1 ? i : Float64(i) for i in rand_int(r, 2^15, 1, 2^16)], pool, SplitMix(103)
end
function dict_any(ks, pool, _)
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
bench(dict_any, dict_any_setup; ops = 2^15, reset = (ks, pool, r) -> shuffle_det!(r, ks))

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

# IdDict lives in the C runtime (eqtable), keyed by object identity. The objects are
# reshuffled every sample; the lookups box `i`, which accounts for most allocations.
iddict_setup() = ([Ref(i) for i in 1:2^15], SplitMix(104))
function iddict(objs, _)
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
bench(iddict, iddict_setup; ops = 2^15, reset = (objs, r) -> shuffle_det!(r, objs))

function set_ops_setup()
    r = SplitMix(4)
    return rand_int(r, 2^16, 1, 2^17), rand_int(r, 2^16, 1, 2^17)
end
function set_ops(a, b)
    sa, sb = Set(a), Set(b)
    return union(sa, sb), intersect(sa, sb), setdiff(sa, sb), unique(a)
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
findall_tuples(v) = findall(has_one, v)
bench(findall_tuples, findall_tuples_setup; ops = 2^16)

# Reductions with init=, which take the mapfoldl path, including over zip with a view
# (#47216, #48792, #49763)
reduce_init_setup() = (v = rand_f64(SplitMix(39), 2^17); (v, view(v, :)))
reduce_init(v, w) = reduce(max, v; init = 0.0) + sum(x -> 4 / (4x - 3), v; init = 0.0) +
                    mapfoldl(t -> t[1] * t[2], +, zip(v, w); init = 0.0)
bench(reduce_init, reduce_init_setup; ops = 3 * 2^17)

# collect and map whose element type is only known at runtime: the eltype widens partway
# through (setindex_widen_up_to). All but the first few elements are reshuffled every
# sample, so the widening happens at the same place.
function collect_widen_setup()
    r = SplitMix(40)
    return Any[1, 2, 1.5, (k == 1 ? 1.5 : k for k in rand_int(r, 2^16, 1, 4))...], SplitMix(140)
end
function collect_widen(xs, _)
    a = [x + 1 for x in xs]
    b = map(x -> x isa Int ? nothing : x, xs)
    return a, b
end
bench(collect_widen, collect_widen_setup; ops = 2 * 2^16, tolerance = 0.06,
      reset = (xs, r) -> shuffle_det!(r, view(xs, 4:length(xs))))

# Arrays as keys and hashed values: a Dict{Vector{Int}} filled and then queried with
# equal copies (hash and isequal of arrays), and hash of Float64 vectors, mostly short,
# a few long
function hash_arrays_setup()
    r = SplitMix(61)
    ks = [rand_int(r, n, 0, 100) for n in rand_int(r, 2048, 1, 32)]
    fs = [rand_f64(r, n) for n in [rand_int(r, 1024, 1, 64); fill(2^14, 4)]]
    return ks, map(copy, ks), fs
end
function hash_arrays(ks, qs, fs)
    d = Dict{Vector{Int},Int}()
    for (i, k) in enumerate(ks)
        d[k] = i
    end
    h = UInt(0)
    for q in qs
        h += get(d, q, 0)
    end
    for f in fs
        h += hash(f)
    end
    return d, h
end
bench(hash_arrays, hash_arrays_setup; ops = 2 * 2048 + 1028)

# Iterating a Dict (the scan over slots), its keys and values, then merge and filter,
# which build new Dicts
dict_iterate_setup() = Dict(zip(rand_int(SplitMix(62), 2^15, 1, 2^40), rand_f64(SplitMix(63), 2^15)))
function dict_iterate(d)
    s = 0.0
    for (k, v) in d
        s += v * (k & 1)
    end
    s += sum(values(d)) + sum(keys(d)) % 7
    return s, merge(d, Dict(1 => 2.0)), filter(p -> p.second > 0.5, d)
end
bench(dict_iterate, dict_iterate_setup; ops = 5 * 2^15)

# Composite keys: Tuple{Int,Symbol} hashed field by field, get! with a default function,
# haskey for misses and pop! with a default
function dict_tuple_keys_setup()
    r = SplitMix(64)
    syms = [Symbol("field_", i) for i in 1:64]
    return [(a, syms[b]) for (a, b) in zip(rand_int(r, 2^16, 1, 2^12), rand_int(r, 2^16, 1, 64))]
end
function dict_tuple_keys(ks)
    d = Dict{Tuple{Int,Symbol},Vector{Int}}()
    for (i, k) in enumerate(ks)
        push!(get!(() -> Int[], d, k), i)
    end
    n = 0
    for k in ks
        n += haskey(d, (k[1] + 2^12, k[2]))
    end
    for i in 1:2:length(ks)
        n += length(pop!(d, ks[i], Int[]))
    end
    return d, n
end
bench(dict_tuple_keys, dict_tuple_keys_setup; ops = 2^16 + 2^16 + 2^15)

# Iterators with nested or stateful iteration: product of three ranges, partition into
# views, and takewhile, whose predicate first fails near the end
function iterators_nested_setup()
    x = rand_int(SplitMix(65), 2^18, 0, 999)
    x[end-100] = 1000
    return x
end
function iterators_nested(x)
    s = 0
    for (i, j, k) in Iterators.product(1:64, 1:64, 1:64)
        s += i * j - k
    end
    for c in Iterators.partition(x, 64)
        s += maximum(c)
    end
    s += sum(Iterators.takewhile(<(1000), x))
    return s
end
bench(iterators_nested, iterators_nested_setup; ops = 64^3 + 2 * 2^18)

# BitSet, which the compiler uses throughout: push!, `in`, union, intersect! and
# iteration over sets with gaps
function bitset_ops_setup()
    r = SplitMix(66)
    return rand_int(r, 2^15, 0, 2^17), rand_int(r, 2^15, 2^10, 2^17 + 2^10)
end
function bitset_ops(a, b)
    sa = BitSet()
    for x in a
        push!(sa, x)
    end
    sb = BitSet(b)
    n = 0
    for x in b
        n += x in sa
    end
    u = union(sa, sb)
    intersect!(sb, sa)
    for x in sb
        n += x
    end
    return u, sb, n
end
bench(bitset_ops, bitset_ops_setup; ops = 3 * 2^15)

end
