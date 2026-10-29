# Arrays: the indexing machinery (Cartesian, views, reinterpret, permuted), bounds-check
# elimination, Memory-backed growth, and Base's sorting and searching.
module ArrayBench
using ..Driver

# Seven-point stencil over CartesianIndices, with bounds checks left in for the
# compiler to prove away
function stencil!(out, u)
    R = CartesianIndices(u)
    I1, Il = first(R), last(R)
    e1, e2, e3 = CartesianIndex(1, 0, 0), CartesianIndex(0, 1, 0), CartesianIndex(0, 0, 1)
    for I in CartesianIndices((I1[1]+1:Il[1]-1, I1[2]+1:Il[2]-1, I1[3]+1:Il[3]-1))
        out[I] = u[I-e1] + u[I+e1] + u[I-e2] + u[I+e2] + u[I-e3] + u[I+e3] - 6u[I]
    end
    return out
end
function stencil_input()
    buf, (out, u) = carve(Float64, 64^3, 2)
    u .= rand_f64(SplitMix(1), 64^3)
    return buf, reshape(out, 64, 64, 64), reshape(u, 64, 64, 64)
end
bench("stencil_3d_cartesian"; ops = 62^3, setup = stencil_input) do (_, out, u)
    stencil!(out, u)
end

# Hand-written matmul in i-k-j order: LICM, the inner loop vectorized
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
function matmul_input()
    r = SplitMix(2)
    buf, (c, a, b) = carve(Float64, 96^2, 3)
    a .= rand_f64(r, 96^2)
    b .= rand_f64(r, 96^2)
    return buf, reshape(c, 96, 96), reshape(a, 96, 96), reshape(b, 96, 96)
end
bench("matmul_loops"; ops = 96^3, setup = matmul_input) do (_, C, A, B)
    matmul!(C, A, B)
end

# Non-contiguous views: SubArray index computation with a stride and a column range
bench("view_strided_sum"; ops = 1024 * 512, setup = () -> reshape(rand_f64(SplitMix(3), 2048 * 1024), 2048, 1024)) do A
    sum(view(A, 1:2:2048, 257:768)) + sum(view(A, 3, :))
end

# ReinterpretArray, historically a performance trap
bench("reinterpret_sum"; ops = 2^18, setup = () -> rand_f32(SplitMix(4), 2^18)) do x
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

function permute_input()
    n = 128 * 64 * 32
    buf, (a, b) = carve(Float64, n, 2)
    a .= rand_f64(SplitMix(5), n)
    return buf, reshape(a, 128, 64, 32), reshape(b, 32, 128, 64)
end
bench("permutedims_3d"; ops = 128 * 64 * 32, setup = permute_input) do (_, A, B)
    permutedims!(B, A, (3, 1, 2))
end

# Growth through push!, i.e. Memory reallocation and the growend path. Reallocation
# goes through the system allocator and page faults, hence the wider tolerance.
bench("push_grow"; ops = 2^18, tolerance = 0.1) do _
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

# The input is restored before every sample: (data, pristine copy)
bench("sort_f64"; ops = 2^18, setup = () -> (x = rand_f64(SplitMix(6), 2^18); (copy(x), x)),
      reset = ((v, x),) -> copyto!(v, x)) do (v, _)
    sort!(v)
end

# Integers with a small range take the counting/radix paths
bench("sort_int"; ops = 2^18, setup = () -> (x = rand_int(SplitMix(7), 2^18, -10^6, 10^6); (copy(x), x)),
      reset = ((v, x),) -> copyto!(v, x)) do (v, _)
    sort!(v)
end

# A by= transform over tuples: comparator inlining through the sort machinery
function sortby_input()
    r = SplitMix(8)
    return collect(zip(rand_int(r, 2^16, 1, 1000), rand_f64(r, 2^16)))
end
bench("sort_by_tuple"; ops = 2^16, setup = () -> (x = sortby_input(); (copy(x), x)),
      reset = ((v, x),) -> copyto!(v, x)) do (v, _)
    sort!(v; by = last, rev = true)
end

# Random queries keep the binary search's branches unpredictable
function search_input()
    r = SplitMix(9)
    return sort!(rand_f64(r, 2^20)), rand_f64(r, 2^14)
end
bench("searchsorted"; ops = 2^14, setup = search_input) do (hay, q)
    s = 0
    for x in q
        s += searchsortedfirst(hay, x)
    end
    s
end

function bits_input()
    r = SplitMix(10)
    return BitVector(isodd.(rand_int(r, 2^20, 0, 1))), BitVector(isodd.(rand_int(r, 2^20, 0, 1)))
end
bench("bitarray_ops"; ops = 2^20, setup = bits_input) do (a, b)
    c = a .& .!b
    count(c) + count(a .⊻ b) + length(findall(c))
end

# Concatenation of many pieces, including splatting a runtime-length vector
function cat_input()
    r = SplitMix(11)
    return [rand_f64(r, 64) for _ in 1:1024]
end
bench("cat_splat"; ops = 1024, setup = cat_input) do vs
    a = reduce(vcat, vs)
    b = hcat(vs...)
    c = vcat(vs...)
    length(a) + size(b, 2) + length(c)
end

end
