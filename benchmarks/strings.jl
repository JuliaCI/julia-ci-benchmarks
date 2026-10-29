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

function csv_input()
    r = SplitMix(1)
    rows = String[]
    for i in 1:4096
        push!(rows, join((string(i), string(rand_f64(r, 1, -1e3, 1e3)[1]), "name_$(i % 97)", string(rand_int(r, 1, 0, 10^6)[1])), ','))
    end
    return rows
end

bench("split_join"; ops = 4096, setup = csv_input) do rows
    n = 0
    for row in rows
        parts = split(row, ',')
        n += length(join(reverse(parts), ';'))
    end
    n
end

bench("interpolate"; ops = 4096, setup = () -> (rand_int(SplitMix(2), 4096, 0, 10^9), rand_f64(SplitMix(3), 4096))) do (ints, floats)
    n = 0
    for i in eachindex(ints)
        s = "item $(ints[i]) of $(i): value=$(floats[i]) [$(isodd(i) ? "odd" : "even")]"
        n += sizeof(s)
    end
    n
end

function numbers_input()
    r = SplitMix(4)
    ints = string.(rand_int(r, 8192, -10^12, 10^12))
    floats = string.(rand_f64(r, 8192, -1e6, 1e6) .* exp.(rand_f64(r, 8192, -20.0, 20.0)))
    return ints, floats
end
bench("parse_numbers"; ops = 16384, setup = numbers_input) do (ints, floats)
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

# Shortest round-trip float printing (Ryu) and integer printing
function print_input()
    r = SplitMix(5)
    return rand_f64(r, 8192, -1e6, 1e6) .* exp.(rand_f64(r, 8192, -20.0, 20.0)), rand_int(r, 8192, -10^15, 10^15)
end
bench("print_numbers"; ops = 16384, setup = print_input) do (fs, is)
    n = 0
    for v in fs
        n += sizeof(string(v))
    end
    for v in is
        n += sizeof(string(v))
    end
    n
end

# PCRE with JIT, captures turned into SubStrings
bench("regex_captures"; ops = 4096, setup = log_input) do lines
    re = r"^(\d{4})-(\d\d)-(\d\d)T[\d:]+ (ERROR|WARN) \[worker-(\d+)\] took (\d+) ms$"
    s = 0
    for l in lines
        m = match(re, l)
        m === nothing && continue
        s += parse(Int, m[6]) + length(m[4])
    end
    s
end

# Non-ASCII text: character iteration, classification and case mapping (utf8proc)
const TEXT = repeat("Größenwahn naïve café Ωμέγα 日本語のテキスト emoji 🎉 and plain ASCII words. ", 400)
bench("unicode_iterate"; ops = length(TEXT), setup = () -> TEXT) do s
    n = 0
    for c in s
        n += isletter(c) + isuppercase(c) + (c == ' ')
    end
    n + length(uppercase(s)) + length(s)
end

bench("replace_search"; ops = 1, setup = () -> join(log_input(), '\n')) do s
    t = replace(s, "ERROR" => "E", "WARN" => "W", r"worker-(\d+)" => s"w\1")
    occursin("took 9999 ms", t) + length(findall("ms\n", t)) + sizeof(t)
end

# The symbol table is a lock-protected hash table in the C runtime
bench("symbol_intern"; ops = 8192, setup = () -> ["sym_$(i % 3000)_$(i % 7)" for i in 1:8192]) do strs
    n = 0
    for s in strs
        n += sizeof(String(Symbol(s)))
    end
    n
end

end
