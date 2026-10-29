# Scalar math: Julia's own libm, integer lowering, and the conversions LLVM must not get
# wrong. Inputs span each function's typical range, with at least 2^16 values per sample
# so the branch predictor cannot learn the range-reduction paths.
module ScalarBench
using ..Driver

const N = 2^16

function sum_map(f, x)
    s = zero(eltype(x))
    @inbounds for v in x
        s += f(v)
    end
    return s
end

bench("exp_f64"; ops = N, setup = () -> rand_f64(SplitMix(1), N, -50.0, 50.0)) do x
    sum_map(exp, x)
end

# log-uniform over 12 decades
bench("log_f64"; ops = N, setup = () -> exp.(rand_f64(SplitMix(2), N, -14.0, 14.0))) do x
    sum_map(log, x)
end

# Mostly the medium reduction path, with every eighth argument large
function trig_input()
    x = rand_f64(SplitMix(3), N, -1e3, 1e3)
    x[8:8:end] .*= 1e6
    return x
end
bench("sincos_f64"; ops = N, setup = trig_input) do x
    s = 0.0
    @inbounds for v in x
        a, b = sincos(v)
        s += a * b
    end
    s
end

function pow_input()
    r = SplitMix(4)
    buf, (xs, ys) = carve(Float64, N, 2)
    xs .= exp.(rand_f64(r, N, -3.0, 3.0))
    ys .= rand_f64(r, N, -8.0, 8.0)
    return buf, xs, ys
end
bench("pow_f64"; ops = N, setup = pow_input) do (_, xs, ys)
    s = 0.0
    @inbounds for i in eachindex(xs)
        s += xs[i]^ys[i]
    end
    s
end

# The vectorizable form: libm inlined into a map! that the loop vectorizer can widen
function exp32_input()
    buf, (x, y) = carve(Float32, 4N, 2)
    x .= rand_f32(SplitMix(5), 4N, -20.0, 20.0)
    return buf, x, y
end
bench("exp_f32_map"; ops = 4N, setup = exp32_input) do (_, x, y)
    map!(exp, y, x)
end

function divrem_input()
    r = SplitMix(6)
    buf, (a, d) = carve(Int, N, 2)
    a .= rand_int(r, N, -10^12, 10^12)
    d .= replace!(rand_int(r, N, -10^6, 10^6), 0 => 7)
    return buf, a, d
end
bench("int_div_rem"; ops = N, setup = divrem_input) do (_, a, d)
    s = 0
    @inbounds for i in eachindex(a)
        s += div(a[i], d[i]) + rem(a[i], d[i]) + fld(a[i], d[i]) + mod(a[i], d[i])
    end
    s
end

function int128_input()
    r = SplitMix(7)
    buf, (a, d) = carve(Int128, N, 2)
    a .= Int128.(rand_int(r, N, -10^15, 10^15)) .* Int128.(rand_int(r, N, -10^15, 10^15))
    d .= Int128.(replace!(rand_int(r, N, -10^9, 10^9), 0 => 3))
    return buf, a, d
end
bench("int128_mul_div"; ops = N, setup = int128_input) do (_, a, d)
    s = Int128(0)
    @inbounds for i in eachindex(a)
        s += (a[i] * 3) ÷ d[i]
    end
    s
end

# Native half precision on aarch64, emulated through Float32 elsewhere
function f16_input()
    r = SplitMix(8)
    buf, (a, b, c) = carve(Float16, 16N, 3)
    a .= Float16.(rand_f64(r, 16N, -2.0, 2.0))
    b .= Float16.(rand_f64(r, 16N, -2.0, 2.0))
    return buf, a, b, c
end
bench("float16_muladd"; ops = 16N, setup = f16_input) do (_, a, b, c)
    @inbounds for i in eachindex(a)
        c[i] = muladd(a[i], b[i], c[i] * Float16(0.5))
    end
    c
end

# Overflow-checked integer arithmetic and checked float-to-int conversion
function checked_input()
    r = SplitMix(9)
    buf, (xi, xf) = carve(N, Int, Float64)
    xi .= rand_int(r, N, -10^9, 10^9)
    xf .= rand_f64(r, N, -1e9, 1e9)
    return buf, xi, xf
end
bench("checked_convert"; ops = N, setup = checked_input) do (_, xi, xf)
    s = 0
    @inbounds for i in eachindex(xi)
        s = Base.checked_add(s, Base.checked_mul(xi[i], 3))
        s = Base.checked_sub(s, round(Int, xf[i]))
        s = Base.checked_add(s, Int(trunc(Int32, xf[i] / 8)))
    end
    s
end

end
