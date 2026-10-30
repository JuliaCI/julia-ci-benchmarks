# Strings and text: splitting, interpolation, number parsing and printing, regexes,
# Unicode iteration and the symbol table.
module StringsBench
using ..Driver
using Printf

# Shared by the benchmarks below: paste this part first to run one of them in the REPL.
function log_input()
    r = SplitMix(6)
    levels = ("INFO", "WARN", "ERROR", "DEBUG")
    return ["2026-10-$(lpad(1 + i % 28, 2, '0'))T12:$(lpad(i % 60, 2, '0')):00 $(levels[1 + i % 4]) [worker-$(i % 13)] took $(rand_int(r, 1, 1, 9999)[1]) ms" for i in 1:4096]
end
# End of the shared part.

function split_join_setup()
    r = SplitMix(1)
    rows = String[]
    for i in 1:4096
        push!(rows, join((string(i), string(rand_f64(r, 1, -1e3, 1e3)[1]), "name_$(i % 97)", string(rand_int(r, 1, 0, 10^6)[1])), ','))
    end
    return rows, similar(rows)
end
function split_join(rows, out)
    for (i, row) in enumerate(rows)
        parts = split(row, ',')
        out[i] = join(reverse(parts), ';')
    end
    return out
end
bench(split_join, split_join_setup; ops = 4096)

interpolate_setup() = (rand_int(SplitMix(2), 4096, 0, 10^9), rand_f64(SplitMix(3), 4096), Vector{String}(undef, 4096))
function interpolate(ints, floats, out)
    for i in eachindex(ints)
        out[i] = "item $(ints[i]) of $(i): value=$(floats[i]) [$(isodd(i) ? "odd" : "even")]"
    end
    return out
end
bench(interpolate, interpolate_setup; ops = 4096)

function parse_numbers_setup()
    r = SplitMix(4)
    ints = string.(rand_int(r, 8192, -10^12, 10^12))
    floats = string.(rand_f64(r, 8192, -1e6, 1e6) .* exp.(rand_f64(r, 8192, -20.0, 20.0)))
    return ints, floats
end
function parse_numbers(ints, floats)
    s = 0
    t = 0.0
    for v in ints
        s += parse(Int, v)
    end
    for v in floats
        t += parse(Float64, v)
    end
    s, t
end
bench(parse_numbers, parse_numbers_setup; ops = 16384)

# Shortest round-trip float printing (Ryu) and integer printing
function print_numbers_setup()
    r = SplitMix(5)
    return rand_f64(r, 8192, -1e6, 1e6) .* exp.(rand_f64(r, 8192, -20.0, 20.0)), rand_int(r, 8192, -10^15, 10^15),
           Vector{String}(undef, 16384)
end
function print_numbers(fs, is, out)
    for (i, v) in enumerate(fs)
        out[i] = string(v)
    end
    for (i, v) in enumerate(is)
        out[8192+i] = string(v)
    end
    return out
end
bench(print_numbers, print_numbers_setup; ops = 16384)

# PCRE with JIT, captures turned into SubStrings. The lines are reshuffled every sample,
# so whether a line matches is not a repeating pattern.
regex_captures_setup() = (log_input(), SplitMix(106))
function regex_captures(lines, _)
    re = r"^(\d{4})-(\d\d)-(\d\d)T[\d:]+ (ERROR|WARN) \[worker-(\d+)\] took (\d+) ms$"
    s = 0
    for l in lines
        m = match(re, l)
        m === nothing && continue
        s += parse(Int, m[6]) + length(m[4])
    end
    s
end
bench(regex_captures, regex_captures_setup; ops = 4096, reset = (lines, r) -> shuffle_det!(r, lines))

# Non-ASCII text: character iteration, classification and case mapping (utf8proc). Words
# from ASCII, Latin-1, Greek, CJK and emoji, so 1- to 4-byte characters interleave.
function unicode_iterate_setup()
    r = SplitMix(7)
    plain = collect("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,")
    alphabets = (plain, plain, collect("àáâäçèéêëìíîïñòóôöùúûüÿßÀÉÖÜ"), collect("αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΣΩ"),
                 collect("日本語中文字漢テキストのをは書読"), collect("🎉😀🚀🌍✨🐍🍕🎵"))
    cs = Char[]
    while length(cs) < 2^16
        a = alphabets[rand_int(r, 1, 1, 6)[1]]
        append!(cs, a[rand_int(r, rand_int(r, 1, 1, 10)[1], 1, length(a))])
        push!(cs, ' ')
    end
    return String(cs[1:2^16])
end
function unicode_iterate(s)
    n = 0
    for c in s
        n += isletter(c) + isuppercase(c) + (c == ' ')
    end
    return n + length(s), uppercase(s)
end
bench(unicode_iterate, unicode_iterate_setup; ops = 2^16)

replace_search_setup() = join(log_input(), '\n')
function replace_search(s)
    t = replace(s, "ERROR" => "E", "WARN" => "W", r"worker-(\d+)" => s"w\1")
    return t, findall("ms\n", t), occursin("took 9999 ms", t)
end
bench(replace_search, replace_search_setup; ops = 1)

# The symbol table is a lock-protected hash table in the C runtime. The warmup calls
# intern every name, so samples measure lookups of existing symbols.
symbol_intern_setup() = (["sym_$(i % 3000)_$(i % 7)" for i in 1:8192], Vector{Symbol}(undef, 8192))
function symbol_intern(strs, out)
    for (i, s) in enumerate(strs)
        out[i] = Symbol(s)
    end
    return out
end
bench(symbol_intern, symbol_intern_setup; ops = 8192)


# Small strings from integers: string(k), other bases, padding, mixed concatenation
# (#53950)
small_strings_setup() = (rand_int(SplitMix(41), 2^14, -10^6, 10^6), Vector{String}(undef, 4 * 2^14))
function small_strings(ks, out)
    for (i, k) in enumerate(ks)
        j = 4(i - 1)
        out[j+1] = string(k)
        out[j+2] = string(k; base = 16)
        out[j+3] = lpad(k, 9, '0')
        out[j+4] = string("x", k, '_', k % 7)
    end
    return out
end
bench(small_strings, small_strings_setup; ops = 4 * 2^14)

# UTF-8 validation and counting over long strings: isvalid on bytes (a shift-based DFA),
# isascii on ASCII text with one non-ASCII character at the end, and length
function utf8_validate_setup()
    r = SplitMix(67)
    alphabets = (collect("abcdefghijklmnopqrstuvwxyz ,."), collect("àáâäçèéêëñòóöùúüß"),
                 collect("αβγδεζηθικλμνξοπρστυφχψω"), collect("日本語中文字漢テキスト"), collect("🎉😀🚀🌍✨"))
    cs = Char[]
    while length(cs) < 2^16
        a = alphabets[rand_int(r, 1, 1, 5)[1]]
        append!(cs, a[rand_int(r, rand_int(r, 1, 1, 12)[1], 1, length(a))])
    end
    mixed = String(cs)
    ascii = String(UInt8.(rand_int(r, 2^18, 0x20, 0x7e))) * "é"
    return Vector{UInt8}(mixed), mixed, ascii
end
utf8_validate(b, mixed, ascii) = (isvalid(String, b), isascii(ascii), length(mixed), length(ascii))
bench(utf8_validate, utf8_validate_setup; ops = 3 * 2^18)

# Printf's generated formatting code: padded and hexadecimal integers, fixed, exponent
# and %g floats, strings
function printf_format_setup()
    r = SplitMix(68)
    return rand_int(r, 4096, -10^9, 10^9), rand_f64(r, 4096, -1e4, 1e4), Vector{String}(undef, 4096)
end
function printf_format(is, fs, out)
    for i in eachindex(is, fs, out)
        out[i] = @sprintf("%8d|%08x|%.3f|%10.4e|%-6s|%g", is[i], is[i] & 0xffffff, fs[i], fs[i], "ab", fs[i] * 1e-3)
    end
    return out
end
bench(printf_format, printf_format_setup; ops = 4096)

# Everyday transforms of short strings: strip, lowercase, startswith and endswith, chop,
# replace of a Char, repeat and concatenation. Half the lines are padded, and the lines
# are reshuffled every sample.
function string_transforms_setup()
    lines = [isodd(i) ? string("  ", l, " ") : l for (i, l) in enumerate(log_input())]
    return lines, Vector{String}(undef, length(lines)), SplitMix(168)
end
function string_transforms(lines, out, _)
    for (i, l) in enumerate(lines)
        t = lowercase(strip(l))
        if startswith(t, "2026") && endswith(t, "ms")
            t = chop(t; tail = 3)
        end
        out[i] = replace(t, ' ' => '_') * repeat("-", i % 4) * "|"
    end
    return out
end
bench(string_transforms, string_transforms_setup; ops = 4096, reset = (lines, out, r) -> shuffle_det!(r, lines))

# Path manipulation as Pkg and code loading do it, all string work: splitdir, splitext,
# joinpath, and normpath resolving . and .. components
function path_ops_setup()
    r = SplitMix(69)
    parts = ["src", "test", "..", ".", "deps", "lib", "Foo.jl", "bar_baz", "v1.2.3", "~tmp"]
    paths = [join(parts[rand_int(r, n, 1, 10)], "/") * (n % 3 == 0 ? ".jl" : "") for n in rand_int(r, 4096, 2, 7)]
    return paths, Vector{String}(undef, 4096)
end
function path_ops(paths, out)
    for (i, p) in enumerate(paths)
        d, b = splitdir(p)
        stem, ext = splitext(b)
        out[i] = normpath(joinpath("/home/user", d, string(stem, ext == ".jl" ? ".ji" : ext)))
    end
    return out
end
bench(path_ops, path_ops_setup; ops = 4096)

end
