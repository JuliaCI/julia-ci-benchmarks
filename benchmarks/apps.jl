# Small programs rather than kernels: each runs many of the paths the other categories
# isolate, in the proportions real code has. A regression here that no isolated
# benchmark shows points at an interaction (inlining budgets, register pressure, layout).
module AppsBench
using ..Driver

# N-body with immutable 3-tuples: SROA, tuple arithmetic, sqrt in a tight loop
struct Body
    pos::NTuple{3,Float64}
    vel::NTuple{3,Float64}
    m::Float64
end
function nbody_setup()
    r = SplitMix(1)
    x = [Body(Tuple(rand_f64(r, 3, -1, 1)), Tuple(rand_f64(r, 3, -0.1, 0.1)), rand_f64(r, 1, 0.5, 2)[1]) for _ in 1:32]
    return copy(x), x
end
function nbody_step!(bs::Vector{Body}, dt)
    n = length(bs)
    @inbounds for i in 1:n
        bi = bs[i]
        v = bi.vel
        for j in 1:n
            i == j && continue
            d = bs[j].pos .- bi.pos
            r2 = d[1]^2 + d[2]^2 + d[3]^2 + 1e-9
            v = v .+ d .* (dt * bs[j].m / (r2 * sqrt(r2)))
        end
        bs[i] = Body(bi.pos, v, bi.m)
    end
    @inbounds for i in 1:n
        b = bs[i]
        bs[i] = Body(b.pos .+ dt .* b.vel, b.vel, b.m)
    end
end
function nbody(bs, _)
    for _ in 1:100
        nbody_step!(bs, 0.001)
    end
    bs[1].pos[1]
end
bench(nbody, nbody_setup; ops = 100 * 32^2, reset = (v, x) -> copyto!(v, x))

# Escape-time iteration: data-dependent trip counts, complex arithmetic
function mandel(c::ComplexF64, maxit)
    z = c
    for k in 1:maxit
        abs2(z) > 4 && return k
        z = z * z + c
    end
    return maxit
end
function mandelbrot()
    s = 0
    for j in 1:150, i in 1:200
        s += mandel(complex(-2.2 + 3.0 * i / 200, -1.2 + 2.4 * j / 150), 80)
    end
    s
end
bench(mandelbrot; ops = 200 * 150)

# Spectral norm: an implicitly defined matrix, integer-to-float conversions, dot products
spectralnorm_setup() = (ones(200), zeros(200), zeros(200))
A(i, j) = 1.0 / ((i + j) * (i + j + 1) / 2 + i + 1)
function Av!(y, x)
    n = length(x)
    @inbounds for i in 1:n
        s = 0.0
        for j in 1:n
            s += A(i - 1, j - 1) * x[j]
        end
        y[i] = s
    end
    return y
end
function Atv!(y, x)
    n = length(x)
    @inbounds for i in 1:n
        s = 0.0
        for j in 1:n
            s += A(j - 1, i - 1) * x[j]
        end
        y[i] = s
    end
    return y
end
function spectralnorm(u, v, t)
    fill!(u, 1.0)
    for _ in 1:10
        Atv!(v, Av!(t, u))
        Atv!(u, Av!(t, v))
    end
    sqrt(sum(u .* v) / sum(v .* v))
end
bench(spectralnorm, spectralnorm_setup; ops = 10 * 4 * 200^2)

# Fannkuch: permutations and prefix reversals of small integer arrays, very branchy
# (the benchmarks-game algorithm)
function fannkuch_count(n)
    p = collect(1:n)
    q = similar(p)
    s = collect(1:n)
    sign = 1
    maxflips = 0
    checksum = 0
    while true
        q0 = p[1]
        if q0 != 1
            copyto!(q, p)
            flips = 1
            while true
                qq = q[q0]
                if qq == 1
                    checksum += sign * flips
                    maxflips = max(maxflips, flips)
                    break
                end
                q[q0] = q0
                if q0 >= 4
                    i, j = 2, q0 - 1
                    while i < j
                        q[i], q[j] = q[j], q[i]
                        i += 1
                        j -= 1
                    end
                end
                q0 = qq
                flips += 1
            end
        end
        if sign == 1
            p[2], p[1] = p[1], p[2]
            sign = -1
        else
            p[2], p[3] = p[3], p[2]
            sign = 1
            for i in 3:n
                sx = s[i]
                if sx != 1
                    s[i] = sx - 1
                    break
                end
                i == n && return checksum, maxflips
                s[i] = i
                t = p[1]
                for j in 1:i
                    p[j] = p[j+1]
                end
                p[i+1] = t
            end
        end
    end
end
fannkuch() = fannkuch_count(8)
bench(fannkuch; ops = factorial(8))

# A JSON parser over UInt8s into Dict{String,Any}/Vector{Any}: type-unstable values,
# string building, recursion
json_parse_setup() = Vector{UInt8}(gen_json(SplitMix(2), 10))
function gen_json(r, depth)
    k = Driver.next!(r) % 6
    depth == 0 && (k = k % 3)
    k == 0 && return string(Int(Driver.next!(r) % 100000))
    k == 1 && return string(Driver.unit_f64(r) * 100)
    k == 2 && return "\"s$(Driver.next!(r) % 1000)\\n\""
    k == 3 && return "[" * join((gen_json(r, depth - 1) for _ in 1:4), ",") * "]"
    return "{" * join(("\"k$i\": " * gen_json(r, depth - 1) for i in 1:4), ",") * "}"
end
function jparse(b::Vector{UInt8}, i::Int)
    while b[i] == UInt8(' ')
        i += 1
    end
    c = b[i]
    if c == UInt8('{')
        d = Dict{String,Any}()
        i += 1
        while b[i] != UInt8('}')
            b[i] == UInt8(',') && (i += 1)
            while b[i] == UInt8(' '); i += 1; end
            k, i = jparse(b, i)
            while b[i] != UInt8(':'); i += 1; end
            v, i = jparse(b, i + 1)
            d[k::String] = v
        end
        return d, i + 1
    elseif c == UInt8('[')
        a = Any[]
        i += 1
        while b[i] != UInt8(']')
            b[i] == UInt8(',') && (i += 1)
            v, i = jparse(b, i)
            push!(a, v)
        end
        return a, i + 1
    elseif c == UInt8('"')
        io = IOBuffer()
        i += 1
        while b[i] != UInt8('"')
            if b[i] == UInt8('\\')
                write(io, b[i+1] == UInt8('n') ? UInt8('\n') : b[i+1])
                i += 2
            else
                write(io, b[i])
                i += 1
            end
        end
        return String(take!(io)), i + 1
    else
        j = i
        isfloat = false
        while j <= length(b) && (UInt8('0') <= b[j] <= UInt8('9') || b[j] == UInt8('.') || b[j] == UInt8('-') || b[j] == UInt8('e'))
            isfloat |= b[j] == UInt8('.') || b[j] == UInt8('e')
            j += 1
        end
        s = String(b[i:j-1])
        return (isfloat ? parse(Float64, s) : parse(Int, s)), j
    end
end
function json_parse(b)
    v, _ = jparse(b, 1)
    length(v)
end
bench(json_parse, json_parse_setup)

# A bytecode interpreter: a dispatch loop over opcodes, an operand stack, data-dependent
# jumps. The program counts the primes below 5000 by trial division.
const PUSH, LOAD, STORE, ADD, MUL, MOD, LT, GT, JMP, JZ, JNZ, HALT = 1:12
function assemble(src)
    labels = Dict{Symbol,Int}()
    code = Int[]
    for ins in src
        if ins isa Symbol
            labels[ins] = length(code) + 1
        else
            push!(code, ins[1], 0)
        end
    end
    pc = 1
    for ins in src
        ins isa Symbol && continue
        arg = length(ins) > 1 ? ins[2] : 0
        code[pc+1] = arg isa Symbol ? labels[arg] : arg
        pc += 2
    end
    return code
end
const N_, D_, C_ = 1, 2, 3
const PRIMES = assemble(Any[
    (PUSH, 2), (STORE, N_),
    :loop_n,
    (LOAD, N_), (PUSH, 5000), (LT,), (JZ, :done),
    (PUSH, 2), (STORE, D_),
    :loop_d,
    (LOAD, D_), (LOAD, D_), (MUL,), (LOAD, N_), (GT,), (JNZ, :prime),
    (LOAD, N_), (LOAD, D_), (MOD,), (JZ, :next),
    (LOAD, D_), (PUSH, 1), (ADD,), (STORE, D_), (JMP, :loop_d),
    :prime,
    (LOAD, C_), (PUSH, 1), (ADD,), (STORE, C_),
    :next,
    (LOAD, N_), (PUSH, 1), (ADD,), (STORE, N_), (JMP, :loop_n),
    :done,
    (HALT,),
])
bytecode_interpreter_setup() = PRIMES
function interpret(code::Vector{Int})
    stack = zeros(Int, 16)
    vars = zeros(Int, 4)
    sp = 0
    pc = 1
    while true
        op = code[pc]
        arg = code[pc+1]
        pc += 2
        if op == PUSH
            sp += 1; stack[sp] = arg
        elseif op == LOAD
            sp += 1; stack[sp] = vars[arg]
        elseif op == STORE
            vars[arg] = stack[sp]; sp -= 1
        elseif op == ADD
            stack[sp-1] += stack[sp]; sp -= 1
        elseif op == MUL
            stack[sp-1] *= stack[sp]; sp -= 1
        elseif op == MOD
            stack[sp-1] = rem(stack[sp-1], stack[sp]); sp -= 1
        elseif op == LT
            stack[sp-1] = stack[sp-1] < stack[sp]; sp -= 1
        elseif op == GT
            stack[sp-1] = stack[sp-1] > stack[sp]; sp -= 1
        elseif op == JMP
            pc = arg
        elseif op == JZ
            stack[sp] == 0 && (pc = arg); sp -= 1
        elseif op == JNZ
            stack[sp] != 0 && (pc = arg); sp -= 1
        else
            return vars[C_]
        end
    end
end
bytecode_interpreter(p) = interpret(p)
bench(bytecode_interpreter, bytecode_interpreter_setup)

# LU with partial pivoting, bounds checks left on: bounds-check elimination and
# loop codegen without @inbounds
lu_checked_setup() = (x = reshape(rand_f64(SplitMix(3), 160^2, -1, 1), 160, 160); (copy(x), x))
function lu_nopivot!(A)
    n = size(A, 1)
    for k in 1:n
        p = k
        for i in k+1:n
            abs(A[i, k]) > abs(A[p, k]) && (p = i)
        end
        if p != k
            for j in 1:n
                A[k, j], A[p, j] = A[p, j], A[k, j]
            end
        end
        piv = A[k, k]
        for i in k+1:n
            A[i, k] /= piv
        end
        for j in k+1:n
            a = A[k, j]
            for i in k+1:n
                A[i, j] -= A[i, k] * a
            end
        end
    end
    return A
end
lu_checked(A, _) = lu_nopivot!(A)[end, end]
bench(lu_checked, lu_checked_setup; ops = 160^3 ÷ 3, reset = (v, x) -> copyto!(v, x))

# RK4 on the Lorenz system with tuples, the StaticArrays style without the package
lorenz(u) = (10.0 * (u[2] - u[1]), u[1] * (28.0 - u[3]) - u[2], u[1] * u[2] - 8 / 3 * u[3])
function rk4(u, h, n)
    for _ in 1:n
        k1 = lorenz(u)
        k2 = lorenz(u .+ (h / 2) .* k1)
        k3 = lorenz(u .+ (h / 2) .* k2)
        k4 = lorenz(u .+ h .* k3)
        u = u .+ (h / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
    end
    return u
end
ode_rk4_tuples() = rk4((1.0, 0.0, 0.0), 1e-3, 2^16)
bench(ode_rk4_tuples; ops = 2^16)

# Breadth-first search over a random sparse graph: pointer chasing and a queue
function graph_bfs_setup()
    r = SplitMix(4)
    n = 2^15
    adj = [Int[] for _ in 1:n]
    for (a, b) in zip(rand_int(r, 4n, 1, n), rand_int(r, 4n, 1, n))
        push!(adj[a], b)
        push!(adj[b], a)
    end
    return adj, zeros(Int, n)
end
function graph_bfs(adj, dist)
    fill!(dist, -1)
    q = [1]
    dist[1] = 0
    head = 1
    while head <= length(q)
        v = q[head]
        head += 1
        for w in adj[v]
            if dist[w] < 0
                dist[w] = dist[v] + 1
                push!(q, w)
            end
        end
    end
    sum(dist)
end
bench(graph_bfs, graph_bfs_setup; ops = 2^15)


# Parsing CSV text into typed columns: line splitting, Int and Float64 parsing from
# substrings, a column with missing values, string pooling through a Dict
function csv_parse_setup()
    r = SplitMix(10)
    n = 4096
    ps = rand_f64(r, n, 0, 1000)
    qs = rand_int(r, n, 0, 99)
    cs = rand_int(r, n, 1, 6)
    ms = rand_int(r, n, 1, 10)
    cities = ["Lund", "Malmö", "Göteborg", "Stockholm", "Uppsala", "Umeå"]
    io = IOBuffer()
    println(io, "id,price,qty,city,flag")
    for i in 1:n
        print(io, i, ',', round(ps[i]; digits = 2), ',')
        ms[i] == 1 || print(io, qs[i])
        println(io, ',', cities[cs[i]], ',', isodd(qs[i]) ? "true" : "false")
    end
    return String(take!(io))
end
function csv_parse(s)
    ids = Int[]
    price = Float64[]
    qty = Union{Missing,Int}[]
    city = Int[]
    flag = Bool[]
    pool = Dict{String,Int}()
    for (k, line) in enumerate(eachline(IOBuffer(s)))
        k == 1 && continue
        f = split(line, ',')
        push!(ids, parse(Int, f[1]))
        push!(price, parse(Float64, f[2]))
        push!(qty, isempty(f[3]) ? missing : parse(Int, f[3]))
        push!(city, get!(pool, f[4], length(pool) + 1))
        push!(flag, f[5] == "true")
    end
    return sum(price) + sum(skipmissing(qty)) + count(flag) + length(pool) + city[end]
end
bench(csv_parse, csv_parse_setup; ops = 4096)

# A small ray tracer: 3-vectors as immutable structs, a scene in a Vector of a small
# Union split at each call, shadow rays and recursive reflections
struct V3
    x::Float64
    y::Float64
    z::Float64
end
vadd(a::V3, b::V3) = V3(a.x + b.x, a.y + b.y, a.z + b.z)
vsub(a::V3, b::V3) = V3(a.x - b.x, a.y - b.y, a.z - b.z)
vscale(a::V3, s) = V3(a.x * s, a.y * s, a.z * s)
vdot(a::V3, b::V3) = a.x * b.x + a.y * b.y + a.z * b.z
vnorm(a::V3) = vscale(a, 1 / sqrt(vdot(a, a)))
struct Sphere
    c::V3
    r::Float64
    refl::Float64
end
struct Plane
    p::V3
    n::V3
    refl::Float64
end
function hit(s::Sphere, o::V3, d::V3)
    oc = vsub(o, s.c)
    b = vdot(oc, d)
    disc = b * b - vdot(oc, oc) + s.r^2
    disc < 0 && return Inf
    t = -b - sqrt(disc)
    return t > 1e-6 ? t : Inf
end
function hit(p::Plane, o::V3, d::V3)
    den = vdot(p.n, d)
    abs(den) < 1e-9 && return Inf
    t = vdot(vsub(p.p, o), p.n) / den
    return t > 1e-6 ? t : Inf
end
normal(s::Sphere, x::V3) = vnorm(vsub(x, s.c))
normal(p::Plane, ::V3) = p.n
function nearest(scene, o, d)
    tbest, best = Inf, 0
    for i in eachindex(scene)
        t = hit(scene[i], o, d)
        if t < tbest
            tbest, best = t, i
        end
    end
    return tbest, best
end
function trace(scene, o::V3, d::V3, depth)
    t, i = nearest(scene, o, d)
    i == 0 && return 0.2
    ob = scene[i]
    x = vadd(o, vscale(d, t))
    n = normal(ob, x)
    l = vnorm(V3(1.0, 2.0, -1.5))
    c = 0.1 + max(0.0, vdot(n, l)) * (nearest(scene, x, l)[2] == 0 ? 1.0 : 0.3)
    if depth < 3 && ob.refl > 0
        r = vsub(d, vscale(n, 2 * vdot(d, n)))
        c = (1 - ob.refl) * c + ob.refl * trace(scene, x, r, depth + 1)
    end
    return c
end
function ray_tracer_setup()
    r = SplitMix(11)
    v = rand_f64(r, 5 * 12)
    scene = Union{Sphere,Plane}[Plane(V3(0.0, -1.0, 0.0), V3(0.0, 1.0, 0.0), 0.3)]
    for k in 0:11
        push!(scene, Sphere(V3(4v[5k+1] - 2, 2v[5k+2] - 0.5, 4v[5k+3] + 1), 0.2 + 0.4v[5k+4], v[5k+5] > 0.5 ? 0.5 : 0.0))
    end
    return scene
end
function ray_tracer(scene)
    w, h = 64, 48
    s = 0.0
    for j in 1:h, i in 1:w
        d = vnorm(V3((i - w / 2) / h, (h / 2 - j) / h, 1.0))
        s += trace(scene, V3(0.0, 0.5, -3.0), d, 0)
    end
    return s
end
bench(ray_tracer, ray_tracer_setup; ops = 64 * 48)

# Dijkstra on a random sparse graph in CSR form with a hand-written binary heap of
# (distance, vertex) tuples and lazy deletion
function dijkstra_setup()
    r = SplitMix(12)
    n, deg = 2^13, 6
    offs = collect(1:deg:n*deg+1)
    return offs, rand_int(r, n * deg, 1, n), rand_f64(r, n * deg, 0.1, 10.0), zeros(n), sizehint!(Tuple{Float64,Int}[], n * deg)
end
function heap_push!(h, x)
    push!(h, x)
    i = length(h)
    @inbounds while i > 1
        p = i >> 1
        h[p][1] <= x[1] && break
        h[i] = h[p]
        i = p
    end
    @inbounds h[i] = x
    return h
end
function heap_pop!(h)
    top = h[1]
    x = pop!(h)
    n = length(h)
    if n > 0
        i = 1
        @inbounds while true
            c = 2i
            c > n && break
            c < n && h[c+1][1] < h[c][1] && (c += 1)
            h[c][1] >= x[1] && break
            h[i] = h[c]
            i = c
        end
        @inbounds h[i] = x
    end
    return top
end
function dijkstra(offs, dst, w, dist, heap)
    fill!(dist, Inf)
    empty!(heap)
    dist[1] = 0.0
    heap_push!(heap, (0.0, 1))
    while !isempty(heap)
        d, u = heap_pop!(heap)
        d > dist[u] && continue
        for k in offs[u]:offs[u+1]-1
            v = dst[k]
            nd = d + w[k]
            if nd < dist[v]
                dist[v] = nd
                heap_push!(heap, (nd, v))
            end
        end
    end
    return sum(x -> isfinite(x) ? x : 0.0, dist)
end
bench(dijkstra, dijkstra_setup; ops = 6 * 2^13)

# LZ77-style compression of generated text and decompression back: a hash table of
# recent positions, byte-wise match extension, varint-coded tokens
function lz_setup()
    r = SplitMix(13)
    letters = collect("etaoinshrdlucmfwypvbgk")
    words = [String([letters[k] for k in rand_int(r, l, 1, length(letters))]) for l in rand_int(r, 400, 2, 9)]
    # Skewed word choice, so that some words repeat often and others rarely
    picks = [1 + floor(Int, 400 * u^3) for u in rand_f64(r, 12000)]
    src = Vector{UInt8}(join(words[picks], ' '))
    return src, zeros(Int, 4096), UInt8[], UInt8[]
end
function put_varint!(out, x)
    while x >= 0x80
        push!(out, (x & 0x7f) % UInt8 | 0x80)
        x >>= 7
    end
    push!(out, x % UInt8)
end
function get_varint(buf, i)
    x, s = 0, 0
    while true
        b = buf[i]
        i += 1
        x |= Int(b & 0x7f) << s
        s += 7
        b < 0x80 && return x, i
    end
end
function emit!(out, src, lit, i, len, off)
    put_varint!(out, i - lit)
    append!(out, view(src, lit:i-1))
    put_varint!(out, len)
    len > 0 && push!(out, off % UInt8, (off >> 8) % UInt8)
end
function lz_compress!(out, src, table)
    fill!(table, 0)
    empty!(out)
    n = length(src)
    i = lit = 1
    while i + 3 <= n
        k = UInt32(src[i]) | UInt32(src[i+1]) << 8 | UInt32(src[i+2]) << 16 | UInt32(src[i+3]) << 24
        h = Int((k * 0x9e3779b1) >> 20) + 1
        c = table[h]
        table[h] = i
        if c > 0 && i - c < 65536 && src[c] == src[i] && src[c+1] == src[i+1] && src[c+2] == src[i+2] && src[c+3] == src[i+3]
            len = 4
            while i + len <= n && src[c+len] == src[i+len]
                len += 1
            end
            emit!(out, src, lit, i, len, i - c)
            i += len
            lit = i
        else
            i += 1
        end
    end
    emit!(out, src, lit, n + 1, 0, 0)
    return out
end
function lz_decompress!(dst, buf)
    empty!(dst)
    i = 1
    while i <= length(buf)
        nlit, i = get_varint(buf, i)
        append!(dst, view(buf, i:i+nlit-1))
        i += nlit
        len, i = get_varint(buf, i)
        len == 0 && break
        off = Int(buf[i]) | Int(buf[i+1]) << 8
        i += 2
        start = length(dst) - off
        for j in 1:len
            push!(dst, dst[start+j])
        end
    end
    return dst
end
function lz_roundtrip(src, table, out, dst)
    lz_compress!(out, src, table)
    lz_decompress!(dst, out)
    return length(out) + Int(dst[end])
end
bench(lz_roundtrip, lz_setup; ops = 12000, tolerance = 0.06)

# Group-by aggregation as data-frame code does it: sortperm on a key column, runs of
# equal keys, per-group statistics over views, results in a Dict
function groupby_setup()
    r = SplitMix(14)
    n = 2^15
    return rand_int(r, n, 1, 600), rand_f64(r, n, -1, 1), rand_f64(r, n, 0.5, 2)
end
function groupby_aggregate(keys, vals, wts)
    p = sortperm(keys)
    ks, vs, ws = keys[p], vals[p], wts[p]
    res = Dict{Int,NTuple{3,Float64}}()
    i, n = 1, length(ks)
    while i <= n
        j = i
        while j < n && ks[j+1] == ks[i]
            j += 1
        end
        v = view(vs, i:j)
        w = view(ws, i:j)
        m = sum(v) / length(v)
        res[ks[i]] = (m, sqrt(sum(x -> (x - m)^2, v) / length(v)), sum(v .* w) / sum(w))
        i = j + 1
    end
    return sum(t -> t[1] + t[2] + t[3], values(res))
end
bench(groupby_aggregate, groupby_setup; ops = 2^15)

# SHA-256 over 64 KiB: 32-bit rotates and adds, the message schedule in a small vector,
# eight words of state carried through the rounds
function first_primes(n)
    ps = Int[]
    k = 2
    while length(ps) < n
        all(p -> k % p != 0, ps) && push!(ps, k)
        k += 1
    end
    return ps
end
const SHA_K = [floor(UInt32, (cbrt(p) % 1) * 2.0^32) for p in first_primes(64)]
const SHA_H0 = Tuple(floor(UInt32, (sqrt(p) % 1) * 2.0^32) for p in first_primes(8))
sha256_setup() = (UInt8.(rand_int(SplitMix(15), 2^16, 0, 255)), zeros(UInt32, 64))
rotr(x::UInt32, n) = bitrotate(x, -n)
function sha256_blocks(data, w)
    h = SHA_H0
    for blk in 0:length(data)÷64-1
        @inbounds for t in 1:16
            o = 64blk + 4(t - 1)
            w[t] = UInt32(data[o+1]) << 24 | UInt32(data[o+2]) << 16 | UInt32(data[o+3]) << 8 | UInt32(data[o+4])
        end
        @inbounds for t in 17:64
            s0 = rotr(w[t-15], 7) ⊻ rotr(w[t-15], 18) ⊻ (w[t-15] >> 3)
            s1 = rotr(w[t-2], 17) ⊻ rotr(w[t-2], 19) ⊻ (w[t-2] >> 10)
            w[t] = w[t-16] + s0 + w[t-7] + s1
        end
        a, b, c, d, e, f, g, hh = h
        @inbounds for t in 1:64
            t1 = hh + (rotr(e, 6) ⊻ rotr(e, 11) ⊻ rotr(e, 25)) + ((e & f) ⊻ (~e & g)) + SHA_K[t] + w[t]
            t2 = (rotr(a, 2) ⊻ rotr(a, 13) ⊻ rotr(a, 22)) + ((a & b) ⊻ (a & c) ⊻ (b & c))
            hh, g, f, e, d, c, b, a = g, f, e, d + t1, c, b, a, t1 + t2
        end
        h = h .+ (a, b, c, d, e, f, g, hh)
    end
    return h
end
sha256(data, w) = sha256_blocks(data, w)[1]
bench(sha256, sha256_setup; ops = 2^16)

# Conjugate gradient on a 2-D Poisson problem with a matrix-free 5-point Laplacian:
# a stencil with boundary branches, fused broadcasts and dot products on a few arrays
function cg_poisson_setup()
    n = 96
    x, b, r, p, ap = map(v -> reshape(v, n, n), carve(Float64, n^2, 5))
    b .= reshape(rand_f64(SplitMix(16), n^2, -1, 1), n, n)
    return x, b, r, p, ap
end
function laplace!(y, x)
    n = size(x, 1)
    @inbounds for j in 1:n, i in 1:n
        s = 4x[i, j]
        i > 1 && (s -= x[i-1, j])
        i < n && (s -= x[i+1, j])
        j > 1 && (s -= x[i, j-1])
        j < n && (s -= x[i, j+1])
        y[i, j] = s
    end
    return y
end
function adot(a, b)
    s = 0.0
    @inbounds @simd for i in eachindex(a, b)
        s += a[i] * b[i]
    end
    return s
end
function cg_poisson(x, b, r, p, ap)
    fill!(x, 0.0)
    copyto!(r, b)
    copyto!(p, r)
    rs = adot(r, r)
    for _ in 1:30
        laplace!(ap, p)
        α = rs / adot(p, ap)
        @. x += α * p
        @. r -= α * ap
        rsn = adot(r, r)
        @. p = r + (rsn / rs) * p
        rs = rsn
    end
    return sqrt(rs)
end
bench(cg_poisson, cg_poisson_setup; ops = 30 * 96^2)

end
