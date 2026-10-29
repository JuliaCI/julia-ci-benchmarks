# Vectorization: whether the loop and SLP vectorizers still fire, which depends on the IR
# Julia emits (aliasing metadata, fast-math flags, bounds-check shapes) as much as on LLVM.
# Each of these drops by 2-8x, not a few percent, when vectorization is lost.
module SIMDBench
using ..Driver

const N = 2^16

# Reassociation allowed by @simd: a vectorized, unrolled reduction
bench("sum_f64_simd"; ops = 16N, setup = () -> rand_f64(SplitMix(1), 16N)) do x
    s = 0.0
    @simd for i in eachindex(x)
        @inbounds s += x[i]
    end
    s
end

# Base's own pairwise mapreduce, the path behind sum/prod/maximum on arrays
bench("sum_f32_base"; ops = 16N, setup = () -> rand_f32(SplitMix(2), 16N)) do x
    sum(abs2, x)
end

# NaN- and signed-zero-aware reductions, which only vectorize with care
bench("maximum_f64"; ops = 4N, setup = () -> rand_f64(SplitMix(3), 4N, -1e3, 1e3)) do x
    maximum(x) - minimum(x)
end

function dot_input()
    buf, (x, y) = carve(Float32, 16N, 2)
    copyto!(x, rand_f32(SplitMix(4), 16N))
    copyto!(y, rand_f32(SplitMix(5), 16N))
    return buf, x, y
end
bench("dot_f32_fma"; ops = 16N, setup = dot_input) do (_, x, y)
    s = 0.0f0
    @simd for i in eachindex(x, y)
        @inbounds s = muladd(x[i], y[i], s)
    end
    s
end

# No @simd: two arrays that may alias, so LLVM must emit a runtime overlap check
# before it can vectorize. Lost when Julia's aliasing information regresses.
function axpy!(y, a, x)
    @inbounds for i in eachindex(x, y)
        y[i] += a * x[i]
    end
    return y
end
function axpy_input()
    buf, (y, x) = carve(Float64, 4N, 2)
    copyto!(y, rand_f64(SplitMix(6), 4N))
    copyto!(x, rand_f64(SplitMix(7), 4N))
    return buf, y, x
end
bench("axpy_alias_check"; ops = 4N, setup = axpy_input) do (_, y, x)
    axpy!(y, 1.0000001, x)
end

# A fused broadcast over five arrays; the broadcast machinery must inline completely
function bcast_input()
    r = SplitMix(8)
    buf, vs = carve(Float64, 4N, 5)
    foreach(v -> copyto!(v, rand_f64(r, 4N)), vs)
    return buf, vs
end
bench("broadcast_fused"; ops = 4N, setup = bcast_input) do (_, (y, a, b, c, d))
    y .= 2.0 .* a .+ b .* c .- d ./ 3.0
end

# Broadcasting a matrix against a row and a column needs index extrusion
function bcast2d_input()
    r = SplitMix(9)
    return reshape(rand_f64(r, 512 * 512), 512, 512), rand_f64(r, 512), rand_f64(r, 512)
end
bench("broadcast_2d_extrude"; ops = 512 * 512, setup = bcast2d_input) do (A, col, row)
    A .= A .* col .+ row' .- 1.0
end

# A data-dependent select that must become a vector blend rather than a branch
bench("conditional_select"; ops = 4N, setup = () -> rand_f64(SplitMix(10), 4N, -1.0, 1.0)) do x
    s = 0.0
    @simd for i in eachindex(x)
        @inbounds v = x[i]
        s += ifelse(v > 0, v, -2v)
    end
    s
end

# Narrow integers: sixteen lanes per vector, widening adds
bench("count_bytes"; ops = 16N, setup = () -> UInt8.(rand_int(SplitMix(11), 16N, 0, 255))) do x
    count(>(0x80), x) + Int(sum(x))
end

# Indexed loads that cannot vectorize on most targets: the scalar-loop quality baseline
function gather_input()
    r = SplitMix(12)
    buf1, (x,) = carve(N, Float64)
    x .= rand_f64(r, N)
    return buf1, x, rand_int(r, 4N, 1, N)
end
bench("gather_index"; ops = 4N, setup = gather_input) do (_, x, idx)
    s = 0.0
    @inbounds for i in idx
        s += x[i]
    end
    s
end

# Tuples of four floats: straight-line code the SLP vectorizer should pack
function quat_mul(a::NTuple{4,Float64}, b::NTuple{4,Float64})
    a1, a2, a3, a4 = a
    b1, b2, b3, b4 = b
    return (a1 * b1 - a2 * b2 - a3 * b3 - a4 * b4,
            a1 * b2 + a2 * b1 + a3 * b4 - a4 * b3,
            a1 * b3 - a2 * b4 + a3 * b1 + a4 * b2,
            a1 * b4 + a2 * b3 - a3 * b2 + a4 * b1)
end
function quat_input()
    r = SplitMix(13)
    x = rand_f64(r, 4N, -1.0, 1.0)
    return [(x[i], x[i+1], x[i+2], x[i+3]) for i in 1:4:4N]
end
bench("tuple_slp_quat"; ops = N, setup = quat_input) do q
    acc = (1.0, 0.0, 0.0, 0.0)
    @inbounds for v in q
        acc = quat_mul(acc, v)
        n = 1 / sqrt(acc[1]^2 + acc[2]^2 + acc[3]^2 + acc[4]^2)
        acc = acc .* n
    end
    acc
end

function complex_input()
    r = SplitMix(14)
    buf, (z, w) = carve(ComplexF64, N, 2)
    z .= complex.(rand_f64(r, N, -1, 1), rand_f64(r, N, -1, 1))
    w .= complex.(rand_f64(r, N, -1, 1), rand_f64(r, N, -1, 1))
    return buf, z, w
end
bench("complex_mul_sum"; ops = N, setup = complex_input) do (_, z, w)
    s = zero(ComplexF64)
    @simd for i in eachindex(z, w)
        @inbounds s += z[i] * w[i]
    end
    s
end

end
