# I/O without the disk's noise: IOBuffer printing and reading, Serialization, and one
# file round trip that stays in the page cache (ios_t and libuv paths).
module IOBench
using ..Driver
using Serialization

bench("iobuffer_print"; ops = 8192, setup = () -> (rand_int(SplitMix(1), 8192, -10^9, 10^9), rand_f64(SplitMix(2), 8192))) do (is, fs)
    io = IOBuffer()
    for i in eachindex(is)
        print(io, is[i], '\t', fs[i], '\t', :sym, '\t', "text", '\n')
    end
    position(io)
end

bench("readline_parse"; ops = 8192, setup = () -> String(take!(let io = IOBuffer(); for i in 1:8192; println(io, i, ",", i * 0.5, ",field", i % 10); end; io end))) do s
    io = IOBuffer(s)
    t = 0.0
    for l in eachline(io)
        a, b, _ = split(l, ',')
        t += parse(Int, a) + parse(Float64, b)
    end
    t
end

struct Rec
    id::Int
    name::String
    vals::Vector{Float64}
    tags::Dict{Symbol,Any}
end
function ser_input()
    r = SplitMix(3)
    return [Rec(i, "rec$i", rand_f64(r, 16), Dict{Symbol,Any}(:a => i, :b => "x$i", :c => (i, 1.5))) for i in 1:512]
end
bench("serialize_roundtrip"; ops = 512, setup = ser_input) do recs
    io = IOBuffer()
    serialize(io, recs)
    seekstart(io)
    length(deserialize(io))
end

# 4 MiB written and read back in 64 KiB chunks
function file_input()
    path = tempname()
    return path, UInt8.(rand_int(SplitMix(4), 2^16, 0, 255))
end
bench("file_roundtrip"; ops = 64, setup = file_input, tolerance = 0.15) do (path, chunk)
    open(path, "w") do f
        for _ in 1:64
            write(f, chunk)
        end
    end
    n = 0
    buf = similar(chunk)
    open(path, "r") do f
        while !eof(f)
            readbytes!(f, buf)
            n += buf[1]
        end
    end
    n
end

end
