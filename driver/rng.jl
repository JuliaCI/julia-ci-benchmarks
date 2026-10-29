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
    carve(n, T1, T2, ...) -> (v1::Vector{T1}, v2::Vector{T2}, ...)
    carve(T, n, k)        -> (v1, ..., vk)

Vectors of `n` elements cut from one allocation at fixed offsets. Kernels that stream
several arrays are sensitive to where the arrays lie relative to each other (cache-set
and 4K-aliasing conflicts, worst for power-of-two sizes), and separate allocations land
differently in every process: up to 7x in cycles with identical instructions. The vectors
are zeroed. Their buffer is kept alive by the driver for the life of the process.
"""
function carve(n::Integer, Ts::Type...)
    # Each vector starts on a page boundary plus a different multiple of 192 bytes
    sizes = [cld(n * sizeof(T), 4096) * 4096 + 192 for T in Ts]
    buf = zeros(UInt8, sum(sizes) + 4096)
    skip = (4096 - Int(UInt(pointer(buf)) % 4096)) % 4096
    offsets = cumsum([skip; sizes[1:end-1]])
    push!(CARVED, buf)
    return ntuple(j -> fill!(unsafe_wrap(Array, Ptr{Ts[j]}(pointer(buf, 1 + offsets[j])), n), zero(Ts[j])), length(Ts))
end
const CARVED = Vector{UInt8}[]
carve(::Type{T}, n::Integer, k::Integer) where {T} = carve(n, ntuple(_ -> T, k)...)
