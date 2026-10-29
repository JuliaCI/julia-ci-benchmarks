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
function nbody_input()
    r = SplitMix(1)
    return [Body(Tuple(rand_f64(r, 3, -1, 1)), Tuple(rand_f64(r, 3, -0.1, 0.1)), rand_f64(r, 1, 0.5, 2)[1]) for _ in 1:32]
end
bench("nbody"; ops = 100 * 32^2, setup = () -> (x = nbody_input(); (copy(x), x)),
      reset = ((bs, x),) -> copyto!(bs, x)) do (bs, _)
    for _ in 1:100
        nbody_step!(bs, 0.001)
    end
    bs[1].pos[1]
end

# Escape-time iteration: data-dependent trip counts, complex arithmetic
function mandel(c::ComplexF64, maxit)
    z = c
    for k in 1:maxit
        abs2(z) > 4 && return k
        z = z * z + c
    end
    return maxit
end
bench("mandelbrot"; ops = 200 * 150) do _
    s = 0
    for j in 1:150, i in 1:200
        s += mandel(complex(-2.2 + 3.0 * i / 200, -1.2 + 2.4 * j / 150), 80)
    end
    s
end

# Spectral norm: an implicitly defined matrix, integer-to-float conversions, dot products
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
bench("spectralnorm"; ops = 10 * 4 * 200^2, setup = () -> (ones(200), zeros(200), zeros(200))) do (u, v, t)
    fill!(u, 1.0)
    for _ in 1:10
        Atv!(v, Av!(t, u))
        Atv!(u, Av!(t, v))
    end
    sqrt(sum(u .* v) / sum(v .* v))
end

# Fannkuch: permutations and prefix reversals of small integer arrays, very branchy
# (the benchmarks-game algorithm)
function fannkuch(n)
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
bench("fannkuch"; ops = factorial(8)) do _
    fannkuch(8)
end

# A JSON parser over UInt8s into Dict{String,Any}/Vector{Any}: type-unstable values,
# string building, recursion
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
bench("json_parse"; setup = () -> Vector{UInt8}(gen_json(SplitMix(2), 10))) do b
    v, _ = jparse(b, 1)
    length(v)
end

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
bench("bytecode_interpreter"; setup = () -> PRIMES) do p
    interpret(p)
end

# LU with partial pivoting, bounds checks left on: bounds-check elimination and
# loop codegen without @inbounds
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
lu_input() = (x = reshape(rand_f64(SplitMix(3), 160^2, -1, 1), 160, 160); (copy(x), x))
bench("lu_checked"; ops = 160^3 ÷ 3, setup = lu_input, reset = ((A, x),) -> copyto!(A, x)) do (A, _)
    lu_nopivot!(A)[end, end]
end

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
bench("ode_rk4_tuples"; ops = 2^16) do _
    rk4((1.0, 0.0, 0.0), 1e-3, 2^16)
end

# Breadth-first search over a random sparse graph: pointer chasing and a queue
function graph_input()
    r = SplitMix(4)
    n = 2^15
    adj = [Int[] for _ in 1:n]
    for (a, b) in zip(rand_int(r, 4n, 1, n), rand_int(r, 4n, 1, n))
        push!(adj[a], b)
        push!(adj[b], a)
    end
    return adj, zeros(Int, n)
end
bench("graph_bfs"; ops = 2^15, setup = graph_input) do (adj, dist)
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

end
