# Hash tables and iterator composition: hashing, probing and rehash growth for concrete,
# string and untyped keys, the C-runtime IdDict, and long tuples past the inlining limits.
module CollectionsBench
using ..Driver

const N = 2^16

# Built from empty every sample, so rehash growth is included; half the lookups miss
function dict_int_input()
    r = SplitMix(1)
    return rand_int(r, N, 1, 4N), rand_int(r, N, 1, 4N)
end
bench("dict_int"; ops = 2N, setup = dict_int_input) do (ks, qs)
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

# Word counting: string hashing and comparison
function words_input()
    r = SplitMix(2)
    vocab = [String([Char('a' + Int(Driver.next!(r) % 26)) for _ in 1:rand_int(r, 1, 3, 12)[1]]) for _ in 1:4096]
    return vocab[rand_int(r, N, 1, 4096)]
end
bench("dict_string_count"; ops = N, setup = words_input) do words
    d = Dict{String,Int}()
    for w in words
        d[w] = get(d, w, 0) + 1
    end
    length(d)
end

# Untyped keys and values, as in much of Base and package internals: every hash and
# isequal is a dynamic call
function any_keys_input()
    r = SplitMix(3)
    pool = Any[1, 2.5, "x", :y, (1, 2), 'c', 0x05, 1 // 3, nothing, [1]]
    return Any[i % 3 == 0 ? string(i) : i % 3 == 1 ? i : Float64(i) for i in rand_int(r, N ÷ 2, 1, N)], pool
end
bench("dict_any"; ops = N ÷ 2, setup = any_keys_input) do (ks, pool)
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

# IdDict lives in the C runtime (eqtable), keyed by object identity
function iddict_input()
    return [Ref(i) for i in 1:N÷2]
end
bench("iddict"; ops = N ÷ 2, setup = iddict_input) do objs
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

function sets_input()
    r = SplitMix(4)
    return rand_int(r, N, 1, 2N), rand_int(r, N, 1, 2N)
end
bench("set_ops"; ops = 2N, setup = sets_input) do (a, b)
    sa, sb = Set(a), Set(b)
    length(union(sa, sb)) + length(intersect(sa, sb)) + length(setdiff(sa, sb)) + length(unique(a))
end

# Generators, filters, zip, enumerate and flatten: all must inline to a plain loop
bench("iterator_compose"; ops = 4N, setup = () -> rand_int(SplitMix(5), 4N, -1000, 1000)) do x
    a = sum(v^2 for v in x if isodd(v))
    b = sum(i * v for (i, v) in enumerate(x) if v > 0)
    c = count(p -> p[1] < p[2], zip(x, Iterators.drop(x, 1)))
    d = sum(Iterators.flatten((x, x)))
    a + b + c + d
end

# Tuples past the inference and inlining limits (length > 32): splats, maps and
# reductions over long heterogeneous and homogeneous tuples
bench("long_tuples"; ops = 1000, setup = () -> (ntuple(i -> Float64(i), 40), ntuple(i -> isodd(i) ? i : Float64(i), 36))) do (t, h)
    s = 0.0
    for k in 1:1000
        u = map(x -> x * k, t)
        s += +(u...) + sum(h) + max(t...)
    end
    s
end

end
