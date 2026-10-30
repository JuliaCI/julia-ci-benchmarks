# Strings and text: splitting, interpolation, number parsing and printing, regexes,
# Unicode iteration and the symbol table.
module StringsBench
using ..Driver

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
    return rows
end
function split_join(rows)
    n = 0
    for row in rows
        parts = split(row, ',')
        n += length(join(reverse(parts), ';'))
    end
    n
end
bench(split_join, split_join_setup; ops = 4096)

interpolate_setup() = (rand_int(SplitMix(2), 4096, 0, 10^9), rand_f64(SplitMix(3), 4096))
function interpolate(ints, floats)
    n = 0
    for i in eachindex(ints)
        s = "item $(ints[i]) of $(i): value=$(floats[i]) [$(isodd(i) ? "odd" : "even")]"
        n += sizeof(s)
    end
    n
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
    return rand_f64(r, 8192, -1e6, 1e6) .* exp.(rand_f64(r, 8192, -20.0, 20.0)), rand_int(r, 8192, -10^15, 10^15)
end
function print_numbers(fs, is)
    n = 0
    for v in fs
        n += sizeof(string(v))
    end
    for v in is
        n += sizeof(string(v))
    end
    n
end
bench(print_numbers, print_numbers_setup; ops = 16384)

# PCRE with JIT, captures turned into SubStrings
regex_captures_setup() = log_input()
function regex_captures(lines)
    re = r"^(\d{4})-(\d\d)-(\d\d)T[\d:]+ (ERROR|WARN) \[worker-(\d+)\] took (\d+) ms$"
    s = 0
    for l in lines
        m = match(re, l)
        m === nothing && continue
        s += parse(Int, m[6]) + length(m[4])
    end
    s
end
bench(regex_captures, regex_captures_setup; ops = 4096)

# Non-ASCII text: character iteration, classification and case mapping (utf8proc). Words
# from ASCII, Latin-1, Greek, CJK and emoji, so 1- to 4-byte characters interleave.
function unicode_iterate_setup()
    r = SplitMix(7)
    plain = collect("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,")
    alphabets = (plain, plain, collect("àáâäçèéêëìíîïñòóôöùúûüÿßÀÉÖÜ"), collect("αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΣΩ"),
                 collect("日本語中文字漢テキストのをは書読"), collect("🎉😀🚀🌍✨🐍🍕🎵"))
    cs = Char[]
    while length(cs) < 2^15
        a = alphabets[rand_int(r, 1, 1, 6)[1]]
        append!(cs, a[rand_int(r, rand_int(r, 1, 1, 10)[1], 1, length(a))])
        push!(cs, ' ')
    end
    return String(cs[1:2^15])
end
function unicode_iterate(s)
    n = 0
    for c in s
        n += isletter(c) + isuppercase(c) + (c == ' ')
    end
    n + length(uppercase(s)) + length(s)
end
bench(unicode_iterate, unicode_iterate_setup; ops = 2^15)

replace_search_setup() = join(log_input(), '\n')
function replace_search(s)
    t = replace(s, "ERROR" => "E", "WARN" => "W", r"worker-(\d+)" => s"w\1")
    occursin("took 9999 ms", t) + length(findall("ms\n", t)) + sizeof(t)
end
bench(replace_search, replace_search_setup; ops = 1)

# The symbol table is a lock-protected hash table in the C runtime
symbol_intern_setup() = ["sym_$(i % 3000)_$(i % 7)" for i in 1:8192]
function symbol_intern(strs)
    n = 0
    for s in strs
        n += sizeof(String(Symbol(s)))
    end
    n
end
bench(symbol_intern, symbol_intern_setup; ops = 8192)


# Small strings from integers: string(k), other bases, padding, mixed concatenation
# (#53950)
small_strings_setup() = rand_int(SplitMix(41), 2^14, -10^6, 10^6)
function small_strings(ks)
    n = 0
    for k in ks
        n += sizeof(string(k)) + sizeof(string(k; base = 16)) + sizeof(lpad(k, 9, '0')) +
             sizeof(string("x", k, '_', k % 7))
    end
    return n
end
bench(small_strings, small_strings_setup; ops = 4 * 2^14)

end
