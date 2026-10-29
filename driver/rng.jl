# SplitMix64: inputs must be identical on every julia version, and Base's RNG streams are
# not guaranteed to be stable across versions.
mutable struct SplitMix
    s::UInt64
end

@inline function next!(r::SplitMix)
    r.s += 0x9e3779b97f4a7c15
    z = r.s
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    return z ⊻ (z >> 31)
end

unit_f64(r::SplitMix) = (next!(r) >> 11) * 0x1.0p-53

rand_f64(r::SplitMix, n::Integer, lo::Real = 0.0, hi::Real = 1.0) =
    [lo + (hi - lo) * unit_f64(r) for _ in 1:n]
rand_f32(r::SplitMix, n::Integer, lo::Real = 0.0, hi::Real = 1.0) =
    [Float32(lo + (hi - lo) * unit_f64(r)) for _ in 1:n]
# Uniform in lo:hi (the modulo bias is irrelevant here)
rand_int(r::SplitMix, n::Integer, lo::Integer, hi::Integer) =
    [lo + Int(next!(r) % UInt64(hi - lo + 1)) for _ in 1:n]

function shuffle_det!(r::SplitMix, v::AbstractVector)
    for i in length(v):-1:2
        j = 1 + Int(next!(r) % UInt64(i))
        v[i], v[j] = v[j], v[i]
    end
    return v
end

"""
    carve(T, n, k) -> (buffer, (v1, ..., vk))

`k` vectors of `n` elements of `T`, cut from one allocation at fixed offsets. Kernels
that stream several arrays are sensitive to where the arrays lie relative to each other
(cache-set and 4K-aliasing conflicts), and separate allocations land differently in
every process. Keep `buffer` alive with the vectors (put it in the benchmark state).
"""
function carve(::Type{T}, n::Integer, k::Integer) where {T}
    stride = (cld(n * sizeof(T), 4096) * 4096 + 192) ÷ sizeof(T)
    buf = zeros(T, stride * k + 4096 ÷ sizeof(T))
    base = pointer(buf)
    # Start on a page boundary, whatever alignment the allocator returned
    skip = (4096 - Int(UInt(base) % 4096)) % 4096 ÷ sizeof(T)
    vs = ntuple(j -> unsafe_wrap(Array, pointer(buf, 1 + skip + (j - 1) * stride), n), k)
    return buf, vs
end
