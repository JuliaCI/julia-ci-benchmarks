# Scalar math: Julia's own libm, integer lowering, and the conversions LLVM must not get
# wrong. Inputs span each function's typical range, with at least 2^16 values per sample
# so the branch predictor cannot learn the range-reduction paths.
module ScalarBench
using ..Driver

exp_f64_setup() = rand_f64(SplitMix(1), 2^16, -50.0, 50.0)
function exp_f64(x)
    s = 0.0
    @inbounds for v in x
        s += exp(v)
    end
    s
end
bench(exp_f64, exp_f64_setup; ops = 2^16)

# log-uniform over 12 decades
log_f64_setup() = exp.(rand_f64(SplitMix(2), 2^16, -14.0, 14.0))
function log_f64(x)
    s = 0.0
    @inbounds for v in x
        s += log(v)
    end
    s
end
bench(log_f64, log_f64_setup; ops = 2^16)

# Mostly the medium reduction path, with every eighth argument large
function sincos_f64_setup()
    x = rand_f64(SplitMix(3), 2^16, -1e3, 1e3)
    x[8:8:end] .*= 1e6
    return x
end
function sincos_f64(x)
    s = 0.0
    @inbounds for v in x
        a, b = sincos(v)
        s += a * b
    end
    s
end
bench(sincos_f64, sincos_f64_setup; ops = 2^16)

function pow_f64_setup()
    r = SplitMix(4)
    xs, ys = carve(Float64, 2^16, 2)
    xs .= exp.(rand_f64(r, 2^16, -3.0, 3.0))
    ys .= rand_f64(r, 2^16, -8.0, 8.0)
    return xs, ys
end
function pow_f64(xs, ys)
    s = 0.0
    @inbounds for i in eachindex(xs)
        s += xs[i]^ys[i]
    end
    s
end
bench(pow_f64, pow_f64_setup; ops = 2^16)

# The vectorizable form: libm inlined into a map! that the loop vectorizer can widen
function exp_f32_map_setup()
    x, y = carve(Float32, 2^18, 2)
    x .= rand_f32(SplitMix(5), 2^18, -20.0, 20.0)
    return x, y
end
exp_f32_map(x, y) = map!(exp, y, x)
bench(exp_f32_map, exp_f32_map_setup; ops = 2^18)

function int_div_rem_setup()
    r = SplitMix(6)
    a, d = carve(Int, 2^16, 2)
    a .= rand_int(r, 2^16, -10^12, 10^12)
    d .= replace!(rand_int(r, 2^16, -10^6, 10^6), 0 => 7)
    return a, d
end
function int_div_rem(a, d)
    s = 0
    @inbounds for i in eachindex(a)
        s += div(a[i], d[i]) + rem(a[i], d[i]) + fld(a[i], d[i]) + mod(a[i], d[i])
    end
    s
end
bench(int_div_rem, int_div_rem_setup; ops = 2^16)

function int128_mul_div_setup()
    r = SplitMix(7)
    a, d = carve(Int128, 2^16, 2)
    a .= Int128.(rand_int(r, 2^16, -10^15, 10^15)) .* Int128.(rand_int(r, 2^16, -10^15, 10^15))
    d .= Int128.(replace!(rand_int(r, 2^16, -10^9, 10^9), 0 => 3))
    return a, d
end
function int128_mul_div(a, d)
    s = Int128(0)
    @inbounds for i in eachindex(a)
        s += (a[i] * 3) ÷ d[i]
    end
    s
end
bench(int128_mul_div, int128_mul_div_setup; ops = 2^16)

# Native half precision on aarch64, emulated through Float32 elsewhere
function float16_muladd_setup()
    r = SplitMix(8)
    a, b, c = carve(Float16, 2^20, 3)
    a .= Float16.(rand_f64(r, 2^20, -2.0, 2.0))
    b .= Float16.(rand_f64(r, 2^20, -2.0, 2.0))
    return a, b, c
end
function float16_muladd(a, b, c)
    @inbounds for i in eachindex(a)
        c[i] = muladd(a[i], b[i], c[i] * Float16(0.5))
    end
    c
end
bench(float16_muladd, float16_muladd_setup; ops = 2^20)

# Overflow-checked integer arithmetic and checked float-to-int conversion
function checked_convert_setup()
    r = SplitMix(9)
    xi, xf = carve(2^16, Int, Float64)
    xi .= rand_int(r, 2^16, -10^9, 10^9)
    xf .= rand_f64(r, 2^16, -1e9, 1e9)
    return xi, xf
end
function checked_convert(xi, xf)
    s = 0
    @inbounds for i in eachindex(xi)
        s = Base.checked_add(s, Base.checked_mul(xi[i], 3))
        s = Base.checked_sub(s, round(Int, xf[i]))
        s = Base.checked_add(s, Int(trunc(Int32, xf[i] / 8)))
    end
    s
end
bench(checked_convert, checked_convert_setup; ops = 2^16)

end
