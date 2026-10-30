# Vectorization: whether the loop and SLP vectorizers still fire, which depends on the IR
# Julia emits (aliasing metadata, fast-math flags, bounds-check shapes) as much as on LLVM.
# Each of these drops by 2-8x, not a few percent, when vectorization is lost.
module SIMDBench
using ..Driver

# Reassociation allowed by @simd: a vectorized, unrolled reduction
sum_f64_simd_setup() = rand_f64(SplitMix(1), 2^20)
function sum_f64_simd(x)
    s = 0.0
    @simd for i in eachindex(x)
        @inbounds s += x[i]
    end
    return s
end
bench(sum_f64_simd, sum_f64_simd_setup; ops = 2^20)

# Base's own pairwise mapreduce, the path behind sum/prod/maximum on arrays
sum_f32_base_setup() = rand_f32(SplitMix(2), 2^20)
sum_f32_base(x) = sum(abs2, x)
bench(sum_f32_base, sum_f32_base_setup; ops = 2^20)

# NaN- and signed-zero-aware reductions, which only vectorize with care
maximum_f64_setup() = rand_f64(SplitMix(3), 2^18, -1e3, 1e3)
maximum_f64(x) = maximum(x) - minimum(x)
bench(maximum_f64, maximum_f64_setup; ops = 2^18)

function dot_f32_fma_setup()
    x, y = carve(Float32, 2^20, 2)
    copyto!(x, rand_f32(SplitMix(4), 2^20))
    copyto!(y, rand_f32(SplitMix(5), 2^20))
    return x, y
end
function dot_f32_fma(x, y)
    s = 0.0f0
    @simd for i in eachindex(x, y)
        @inbounds s = muladd(x[i], y[i], s)
    end
    return s
end
bench(dot_f32_fma, dot_f32_fma_setup; ops = 2^20)

# No @simd: two arrays that may alias, so LLVM must emit a runtime overlap check before
# it can vectorize. Lost when Julia's aliasing information regresses.
function axpy_alias_check_setup()
    y, x = carve(Float64, 2^18, 2)
    copyto!(y, rand_f64(SplitMix(6), 2^18))
    copyto!(x, rand_f64(SplitMix(7), 2^18))
    return y, x
end
function axpy_alias_check(y, x)
    @inbounds for i in eachindex(x, y)
        y[i] += 1.0000001 * x[i]
    end
    return y
end
bench(axpy_alias_check, axpy_alias_check_setup; ops = 2^18)

# A fused broadcast over five arrays; the broadcast machinery must inline completely
function broadcast_fused_setup()
    r = SplitMix(8)
    vs = carve(Float64, 2^18, 5)
    foreach(v -> copyto!(v, rand_f64(r, 2^18)), vs)
    return vs
end
broadcast_fused(y, a, b, c, d) = y .= 2.0 .* a .+ b .* c .- d ./ 3.0
bench(broadcast_fused, broadcast_fused_setup; ops = 2^18)

# Broadcasting a matrix against a row and a column needs index extrusion
function broadcast_2d_extrude_setup()
    r = SplitMix(9)
    out, A = carve(Float64, 512 * 512, 2)
    A .= rand_f64(r, 512 * 512)
    return reshape(out, 512, 512), reshape(A, 512, 512), rand_f64(r, 512), rand_f64(r, 512)
end
broadcast_2d_extrude(out, A, col, row) = out .= A .* col .+ row' .- 1.0
bench(broadcast_2d_extrude, broadcast_2d_extrude_setup; ops = 512 * 512)

# A data-dependent select that must become a vector blend rather than a branch
conditional_select_setup() = rand_f64(SplitMix(10), 2^18, -1.0, 1.0)
function conditional_select(x)
    s = 0.0
    @simd for i in eachindex(x)
        @inbounds v = x[i]
        s += ifelse(v > 0, v, -2v)
    end
    return s
end
bench(conditional_select, conditional_select_setup; ops = 2^18)

# Narrow integers: sixteen lanes per vector, widening adds
count_bytes_setup() = UInt8.(rand_int(SplitMix(11), 2^20, 0, 255))
count_bytes(x) = count(>(0x80), x) + Int(sum(x))
bench(count_bytes, count_bytes_setup; ops = 2^20)

# Indexed loads that cannot vectorize on most targets: the scalar-loop quality baseline
function gather_index_setup()
    r = SplitMix(12)
    (x,) = carve(2^16, Float64)
    x .= rand_f64(r, 2^16)
    return x, rand_int(r, 2^18, 1, 2^16)
end
function gather_index(x, idx)
    s = 0.0
    @inbounds for i in idx
        s += x[i]
    end
    return s
end
bench(gather_index, gather_index_setup; ops = 2^18)

# Tuples of four floats: straight-line code the SLP vectorizer should pack
function quat_mul(a::NTuple{4,Float64}, b::NTuple{4,Float64})
    a1, a2, a3, a4 = a
    b1, b2, b3, b4 = b
    return (a1 * b1 - a2 * b2 - a3 * b3 - a4 * b4,
            a1 * b2 + a2 * b1 + a3 * b4 - a4 * b3,
            a1 * b3 - a2 * b4 + a3 * b1 + a4 * b2,
            a1 * b4 + a2 * b3 - a3 * b2 + a4 * b1)
end
function tuple_slp_quat_setup()
    x = rand_f64(SplitMix(13), 2^18, -1.0, 1.0)
    return [(x[i], x[i+1], x[i+2], x[i+3]) for i in 1:4:2^18]
end
function tuple_slp_quat(q)
    acc = (1.0, 0.0, 0.0, 0.0)
    @inbounds for v in q
        acc = quat_mul(acc, v)
        n = 1 / sqrt(acc[1]^2 + acc[2]^2 + acc[3]^2 + acc[4]^2)
        acc = acc .* n
    end
    return acc
end
bench(tuple_slp_quat, tuple_slp_quat_setup; ops = 2^16)

function complex_mul_sum_setup()
    r = SplitMix(14)
    z, w = carve(ComplexF64, 2^16, 2)
    z .= complex.(rand_f64(r, 2^16, -1, 1), rand_f64(r, 2^16, -1, 1))
    w .= complex.(rand_f64(r, 2^16, -1, 1), rand_f64(r, 2^16, -1, 1))
    return z, w
end
function complex_mul_sum(z, w)
    s = zero(ComplexF64)
    @simd for i in eachindex(z, w)
        @inbounds s += z[i] * w[i]
    end
    return s
end
bench(complex_mul_sum, complex_mul_sum_setup; ops = 2^16)


# A float reduction and a Bool all-reduction in one @simd loop (#64384: stopped
# vectorizing with LLVM 23)
function sum_flag_setup()
    w, x = carve(Float32, 2^19, 2)
    r = SplitMix(30)
    w .= rand_f32(r, 2^19)
    x .= rand_f32(r, 2^19, -1, 1)
    return w, x
end
function sum_flag(w, x)
    s = 0.0f0
    ok = true
    @inbounds @simd for i in eachindex(x)
        a = w[i]
        s += a * abs(x[i])
        ok &= floatmin(Float32) <= a <= floatmax(Float32)
    end
    return s, ok
end
bench(sum_flag, sum_flag_setup; ops = 2^19)

# Stride-2 dot products with two accumulators, which SLP or the loop vectorizer may take
# depending on the cost model (#63165)
strided_pairs_setup() = (zeros(128), rand_f64(SplitMix(31), 4096))
pair_coef(i, j) = (i + j - 2) * (i + j - 1) / 2 + i
function strided_pairs(out, v)
    n = length(v)
    @inbounds for i in 1:128
        x1 = x2 = 0.0
        fi = Float64(i)
        for j in 1:2:n
            x1 += v[j] / pair_coef(fi, Float64(j))
            x2 += v[j+1] / pair_coef(fi, Float64(j + 1))
        end
        out[i] = x1 + x2
    end
    return out[1]
end
bench(strided_pairs, strided_pairs_setup; ops = 128 * 4096)

# findmin/findmax and argmin/argmax, which take different paths to the same reduction
find_extrema_setup() = rand_f64(SplitMix(32), 2^16)
find_extrema(x) = findmin(x)[1] + findmax(x)[1] + x[argmin(x)] + x[argmax(x)]
bench(find_extrema, find_extrema_setup; ops = 4 * 2^16)

end
