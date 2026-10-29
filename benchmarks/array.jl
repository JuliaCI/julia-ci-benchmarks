# Arrays: the indexing machinery (Cartesian, views, reinterpret, permuted), bounds-check
# elimination, Memory-backed growth, and Base's sorting and searching.
module ArrayBench
using ..Driver

# Seven-point stencil over CartesianIndices, with bounds checks left in for the
# compiler to prove away
function stencil_3d_cartesian_setup()
    out, u = carve(Float64, 64^3, 2)
    u .= rand_f64(SplitMix(1), 64^3)
    return reshape(out, 64, 64, 64), reshape(u, 64, 64, 64)
end
function stencil!(out, u)
    R = CartesianIndices(u)
    I1, Il = first(R), last(R)
    e1, e2, e3 = CartesianIndex(1, 0, 0), CartesianIndex(0, 1, 0), CartesianIndex(0, 0, 1)
    for I in CartesianIndices((I1[1]+1:Il[1]-1, I1[2]+1:Il[2]-1, I1[3]+1:Il[3]-1))
        out[I] = u[I-e1] + u[I+e1] + u[I-e2] + u[I+e2] + u[I-e3] + u[I+e3] - 6u[I]
    end
    return out
end
stencil_3d_cartesian(out, u) = stencil!(out, u)
bench(stencil_3d_cartesian, stencil_3d_cartesian_setup; ops = 62^3)

# Hand-written matmul in i-k-j order: LICM, the inner loop vectorized
function matmul_loops_setup()
    r = SplitMix(2)
    c, a, b = carve(Float64, 96^2, 3)
    a .= rand_f64(r, 96^2)
    b .= rand_f64(r, 96^2)
    return reshape(c, 96, 96), reshape(a, 96, 96), reshape(b, 96, 96)
end
function matmul!(C, A, B)
    fill!(C, 0)
    n, m = size(A)
    p = size(B, 2)
    @inbounds for j in 1:p, k in 1:m
        b = B[k, j]
        @simd for i in 1:n
            C[i, j] += A[i, k] * b
        end
    end
    return C
end
matmul_loops(C, A, B) = matmul!(C, A, B)
bench(matmul_loops, matmul_loops_setup; ops = 96^3)

# Non-contiguous views: SubArray index computation with a stride and a column range
view_strided_sum_setup() = reshape(rand_f64(SplitMix(3), 2048 * 1024), 2048, 1024)
view_strided_sum(A) = sum(view(A, 1:2:2048, 257:768)) + sum(view(A, 3, :))
bench(view_strided_sum, view_strided_sum_setup; ops = 1024 * 512)

# ReinterpretArray, historically a performance trap
reinterpret_sum_setup() = rand_f32(SplitMix(4), 2^18)
function reinterpret_sum(x)
    s = UInt32(0)
    for v in reinterpret(UInt32, x)
        s += v >> 3
    end
    t = 0.0
    for v in reinterpret(Float64, x)
        t += v
    end
    s, t
end
bench(reinterpret_sum, reinterpret_sum_setup; ops = 2^18)

function permutedims_3d_setup()
    n = 128 * 64 * 32
    a, b = carve(Float64, n, 2)
    a .= rand_f64(SplitMix(5), n)
    return reshape(a, 128, 64, 32), reshape(b, 32, 128, 64)
end
permutedims_3d(A, B) = permutedims!(B, A, (3, 1, 2))
bench(permutedims_3d, permutedims_3d_setup; ops = 128 * 64 * 32)

# Growth through push!, i.e. Memory reallocation and the growend path. Reallocation
# goes through the system allocator and page faults, hence the wider tolerance.
function push_grow()
    v = Int[]
    for i in 1:2^18
        push!(v, i)
    end
    w = Float64[]
    for i in 1:2^16
        pushfirst!(w, i)
        length(w) > 64 && pop!(w)
    end
    v, w
end
bench(push_grow; ops = 2^18, tolerance = 0.1)

# The input is restored before every sample: (data, pristine copy)
sort_f64_setup() = (x = rand_f64(SplitMix(6), 2^18); (copy(x), x))
sort_f64(v, x) = sort!(v)
bench(sort_f64, sort_f64_setup; ops = 2^18, reset = (v, x) -> copyto!(v, x))

# Integers with a small range take the counting/radix paths
sort_int_setup() = (x = rand_int(SplitMix(7), 2^18, -10^6, 10^6); (copy(x), x))
sort_int(v, x) = sort!(v)
bench(sort_int, sort_int_setup; ops = 2^18, reset = (v, x) -> copyto!(v, x))

# A by= transform over tuples: comparator inlining through the sort machinery
function sort_by_tuple_setup()
    r = SplitMix(8)
    x = collect(zip(rand_int(r, 2^16, 1, 1000), rand_f64(r, 2^16)))
    return copy(x), x
end
sort_by_tuple(v, x) = sort!(v; by = last, rev = true)
bench(sort_by_tuple, sort_by_tuple_setup; ops = 2^16, reset = (v, x) -> copyto!(v, x))

# Random queries keep the binary search's branches unpredictable
function searchsorted_random_setup()
    r = SplitMix(9)
    return sort!(rand_f64(r, 2^20)), rand_f64(r, 2^14)
end
function searchsorted_random(hay, q)
    s = 0
    for x in q
        s += searchsortedfirst(hay, x)
    end
    s
end
bench(searchsorted_random, searchsorted_random_setup; ops = 2^14)

function bitarray_ops_setup()
    r = SplitMix(10)
    return BitVector(isodd.(rand_int(r, 2^20, 0, 1))), BitVector(isodd.(rand_int(r, 2^20, 0, 1)))
end
function bitarray_ops(a, b)
    c = a .& .!b
    count(c) + count(a .⊻ b) + length(findall(c))
end
bench(bitarray_ops, bitarray_ops_setup; ops = 2^20)

# Concatenation of many pieces, including splatting a runtime-length vector
function cat_splat_setup()
    r = SplitMix(11)
    return [rand_f64(r, 64) for _ in 1:1024]
end
function cat_splat(vs)
    a = reduce(vcat, vs)
    b = hcat(vs...)
    c = vcat(vs...)
    return a, b, c
end
bench(cat_splat, cat_splat_setup; ops = 1024)

end
