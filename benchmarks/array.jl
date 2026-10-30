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

# Random queries, reshuffled every sample, keep the binary search's branches unpredictable
function searchsorted_random_setup()
    r = SplitMix(9)
    hay, q = carve(Float64 => 2^20, Float64 => 2^14)
    hay .= sort!(rand_f64(r, 2^20))
    q .= rand_f64(r, 2^14)
    return hay, q, SplitMix(109)
end
function searchsorted_random(hay, q, _)
    s = 0
    for x in q
        s += searchsortedfirst(hay, x)
    end
    s
end
bench(searchsorted_random, searchsorted_random_setup; ops = 2^14, reset = (hay, q, r) -> shuffle_det!(r, q))

function bitarray_ops_setup()
    r = SplitMix(10)
    return BitVector(isodd.(rand_int(r, 2^20, 0, 1))), BitVector(isodd.(rand_int(r, 2^20, 0, 1)))
end
function bitarray_ops(a, b)
    c = a .& .!b
    d = a .⊻ b
    f = findall(c)
    return count(c) + count(d) + length(f), c, d, f
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

# Stores into a Union{Missing,Float64} array (payload and type tag written separately),
# then a bulk copy of both. One array in three regions: the first is the input and is
# never written, so no reset is needed.
function union_store_copy_setup()
    b = fill!(Vector{Union{Missing,Float64}}(undef, 3 * 2^18), missing)
    for (i, x) in enumerate(rand_f64(SplitMix(12), 2^18, -1, 1))
        b[i] = x < 0 ? missing : x
    end
    return b
end
function union_store_copy(b)
    n = 2^18
    @inbounds for i in 1:n
        x = b[i]
        b[n + i] = x === missing ? missing : muladd(2.0, x, 1.0)
    end
    copyto!(b, 2n + 1, b, n + 1, n)
    return b
end
bench(union_store_copy, union_store_copy_setup; ops = 2^18)


# Writes through reinterprets whose element size differs from the parent's (Int16 and
# Int32 over bytes): the index arithmetic of ReinterpretArray setindex!, historically
# 10-250x slower than a plain array (#42227, #48801, #51658)
function reinterpret_write_setup()
    b, orig = carve(UInt8, 2^20, 2)
    orig .= UInt8.(rand_int(SplitMix(20), 2^20, 0, 255))
    return copyto!(b, orig), orig
end
zigzag(n) = (n >> one(n)) ⊻ -(n & one(n))
function reinterpret_write(b, _)
    r = reinterpret(Int16, b)
    @inbounds @simd for i in eachindex(r)
        r[i] = zigzag(r[i])
    end
    w = reinterpret(Int32, b)
    w .+= Int32(1)
    return Int(b[1])
end
bench(reinterpret_write, reinterpret_write_setup; ops = 2^20, reset = (b, orig) -> copyto!(b, orig))

# A vector of 3-tuples seen as a 3×n matrix through reinterpret(reshape, ...): column
# access and a reduction over the reshaped reinterpret
function reinterpret_reshape_setup()
    v = rand_f64(SplitMix(21), 3 * 2^15)
    return [(v[3i-2], v[3i-1], v[3i]) for i in 1:2^15]
end
function reinterpret_reshape(pts)
    M = reinterpret(reshape, Float64, pts)
    s = 0.0
    @inbounds for j in axes(M, 2)
        s += M[1, j] * M[2, j] - M[3, j]
    end
    c = sum(M; dims = 2)
    return s + c[1] + c[2] + c[3]
end
bench(reinterpret_reshape, reinterpret_reshape_setup; ops = 2^15)

# 12-byte records packed in a byte buffer and read through reinterpret: each element is
# assembled from several parent elements
struct Rec12
    a::Float32
    b::Float32
    c::Int32
end
reinterpret_records_setup() = UInt8.(rand_int(SplitMix(22), 12 * 2^17, 0, 255))
function reinterpret_records(bytes)
    s = 0
    for r in reinterpret(Rec12, bytes)
        s += r.c + (r.a < r.b)
    end
    return s
end
bench(reinterpret_records, reinterpret_records_setup; ops = 2^17)

# Assignment into 2-D views with UnitRange indices, by broadcast and by copyto!: SubArray
# copyto_unaliased! and in-place broadcast lost vectorization twice in 1.11 (#53158, #53615)
function subarray_assign_setup()
    B, A = carve(Float64 => 257^2, Float64 => 256^2)
    A .= rand_f64(SplitMix(23), 256^2)
    return reshape(B, 257, 257), reshape(A, 256, 256)
end
function subarray_assign(B, A)
    B[1:end-1, 1:end-1] .= A
    copyto!(view(B, 2:size(B, 1), 2:size(B, 2)), A)
    return B
end
bench(subarray_assign, subarray_assign_setup; ops = 2 * 256^2)

# A minimal user array type (IndexLinear, splatted getindex): the AbstractArray fallbacks
# that package arrays rely on for copy, broadcast, reductions and views (#52070)
struct WrapArr{T,N} <: AbstractArray{T,N}
    data::Array{T,N}
end
Base.size(a::WrapArr) = size(a.data)
Base.IndexStyle(::Type{<:WrapArr}) = IndexLinear()
Base.getindex(a::WrapArr, i...) = getindex(a.data, i...)
Base.setindex!(a::WrapArr, v, i...) = setindex!(a.data, v, i...)
Base.similar(a::WrapArr, ::Type{T}, dims::Dims) where {T} = WrapArr(similar(a.data, T, dims))
custom_array_setup() = WrapArr(reshape(rand_f64(SplitMix(24), 2^16), 256, 256))
function custom_array(a)
    b = copy(a)
    c = a .* 2.0 .+ b
    return sum(c) + maximum(view(a, 2:255, 2:255))
end
bench(custom_array, custom_array_setup; ops = 2^16)

# Reductions along leading, inner and trailing dimensions (mapreducedim!)
reduce_dims_setup() = reshape(rand_f64(SplitMix(25), 64 * 64 * 32), 64, 64, 32)
function reduce_dims(A)
    a = sum(A; dims = 1)
    b = sum(abs2, A; dims = 2)
    c = sum(A; dims = 3)
    d = maximum(A; dims = (1, 3))
    return a, b, c, d
end
bench(reduce_dims, reduce_dims_setup; ops = 4 * 64 * 64 * 32)

# Logical indexing with Vector{Bool} and BitVector masks, and findall on a mask
function logical_index_setup()
    r = SplitMix(26)
    x = rand_f64(r, 2^16)
    m = rand_f64(r, 2^16) .< 0.5
    return x, Vector{Bool}(m), BitVector(m)
end
function logical_index(x, m, bm)
    a = x[m]
    b = x[bm]
    return a, b, findall(m), sum(view(x, bm))
end
bench(logical_index, logical_index_setup; ops = 4 * 2^16, tolerance = 0.08)

# Missing values as data-frame code has them: map over two Union{Missing,Float32}
# arrays, skipmissing reductions and filter (#54811, #55374)
function missing_ops_setup()
    r = SplitMix(27)
    f(x) = x < 0.1 ? missing : Float32(x)
    return Union{Missing,Float32}[f(x) for x in rand_f64(r, 2^16)], Union{Missing,Float32}[f(x) for x in rand_f64(r, 2^16)]
end
function missing_ops(a, b)
    c = map(*, a, b)
    return sum(skipmissing(a)) + count(ismissing, c), c, collect(skipmissing(c)), filter(x -> x > 0.5f0, skipmissing(b))
end
bench(missing_ops, missing_ops_setup; ops = 4 * 2^16)

# A deque on a Vector: pushfirst!, insert! near the front, pop! and popfirst!, which grow
# and shrink at both ends (#56026)
front_churn_setup() = rand_int(SplitMix(28), 2^15, 0, 10^6)
function front_churn(xs)
    v = Int[]
    s = 0
    for x in xs
        pushfirst!(v, x)
        insert!(v, 2, x >> 1)
        if length(v) > 96
            s += pop!(v) + popfirst!(v) + pop!(v)
        end
    end
    return s + length(v)
end
bench(front_churn, front_churn_setup; ops = 2^15, tolerance = 0.06)

# Stores through a raw pointer while reading a 3-d array: unless aliasing information
# says otherwise, the array's size and data pointer are reloaded every iteration (#63782)
function ptr_store_setup()
    dst, src = carve(Float64, 5 * 6 * 6000, 2)
    src .= rand_f64(SplitMix(29), length(src))
    return dst, reshape(src, 5, 6, 6000)
end
function ptr_store(dst, src)
    GC.@preserve dst begin
        p = pointer(dst)
        @inbounds for k in axes(src, 3), j in axes(src, 2), i in axes(src, 1)
            idx = i + size(src, 1) * ((j - 1) + size(src, 2) * (k - 1))
            unsafe_store!(p, unsafe_load(p, idx) - 0.5 * src[i, j, k], idx)
        end
    end
    return dst[1]
end
bench(ptr_store, ptr_store_setup; ops = 5 * 6 * 6000)

# Editing a Vector in the middle: deleteat! with sorted indices, insert!, splice!,
# append!, filter! and resize!, which move elements within the Memory
function vector_edit_setup()
    r = SplitMix(57)
    return rand_int(r, 2^14, 0, 10^6), sort!(unique!(rand_int(r, 2^12, 1, 2^14)))
end
function vector_edit(x, idx)
    v = copy(x)
    deleteat!(v, idx)
    for k in 1:256
        insert!(v, 1 + 37k, k)
    end
    splice!(v, 100:612, view(x, 1:256))
    append!(v, x)
    filter!(isodd, v)
    resize!(v, length(v) ÷ 2)
    return v
end
bench(vector_edit, vector_edit_setup; ops = 2^14)

# Slices of a matrix: eachcol and eachrow (a view per column and per row) and mapslices
eachslice_reduce_setup() = reshape(rand_f64(SplitMix(58), 512 * 256), 512, 256)
function eachslice_reduce(A)
    a = map(sum, eachcol(A))
    b = [maximum(r) for r in eachrow(A)]
    c = mapslices(x -> x ./ sum(x), A; dims = 1)
    return a, b, c
end
bench(eachslice_reduce, eachslice_reduce_setup; ops = 3 * 512 * 256)

# Float ranges: StepRangeLen with a TwicePrecision reference and step, and LinRange,
# indexed at random, iterated, and broadcast over
float_ranges_setup() = (rand_int(SplitMix(59), 2^16, 1, 10^6), zeros(2^16))
function float_ranges(idx, out)
    r = range(-1.0, 1.0; length = 10^6)
    l = LinRange(0.0, 3.0, 10^6)
    s = 0.0
    @inbounds for i in idx
        s += r[i] + l[i]
    end
    for v in 0.1:0.1:2000.0
        s += v
    end
    out .= r[1:2^16] .+ 2.0 .* view(l, 1:2^16)
    return s, out
end
bench(float_ranges, float_ranges_setup; ops = 2 * 2^16 + 2 * 10^4)

# Strings sorted: every comparison is a memcmp on variable-length data, many of them with
# a shared prefix. The input is restored before every sample.
function sort_strings_setup()
    r = SplitMix(60)
    pre = ("", "item_", "item_00", "x")
    x = [string(pre[1 + i % 4], n) for (i, n) in enumerate(rand_int(r, 2^14, 0, 10^7))]
    return copy(x), x
end
sort_strings(v, x) = sort!(v)
bench(sort_strings, sort_strings_setup; ops = 2^14, reset = (v, x) -> copyto!(v, x))

end
