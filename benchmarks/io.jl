# I/O without the disk's noise: IOBuffer printing and reading, Serialization, and one
# file round trip that stays in the page cache (ios_t and libuv paths).
module IOBench
using ..Driver
using Serialization

iobuffer_print_setup() = (rand_int(SplitMix(1), 8192, -10^9, 10^9), rand_f64(SplitMix(2), 8192))
function iobuffer_print(is, fs)
    io = IOBuffer()
    for i in eachindex(is)
        # One call per value: print(io, xs...) with mixed types dispatches at runtime
        print(io, is[i])
        print(io, '\t')
        print(io, fs[i])
        print(io, "\t:sym\ttext\n")
    end
    return position(io)
end
bench(iobuffer_print, iobuffer_print_setup; ops = 8192)

function readline_parse_setup()
    io = IOBuffer()
    for i in 1:8192
        println(io, i, ",", i * 0.5, ",field", i % 10)
    end
    return String(take!(io))
end
function readline_parse(s)
    io = IOBuffer(s)
    t = 0.0
    for l in eachline(io)
        a, b, _ = split(l, ',')
        t += parse(Int, a) + parse(Float64, b)
    end
    return t
end
bench(readline_parse, readline_parse_setup; ops = 8192)

struct Rec
    id::Int
    name::String
    vals::Vector{Float64}
    tags::Dict{Symbol,Any}
end
function serialize_roundtrip_setup()
    r = SplitMix(3)
    return [Rec(i, "rec$i", rand_f64(r, 16), Dict{Symbol,Any}(:a => i, :b => "x$i", :c => (i, 1.5))) for i in 1:512]
end
function serialize_roundtrip(recs)
    io = IOBuffer()
    serialize(io, recs)
    seekstart(io)
    return deserialize(io)
end
bench(serialize_roundtrip, serialize_roundtrip_setup; ops = 512)

# 4 MiB written and read back in 64 KiB chunks, then deleted
function file_roundtrip_setup()
    path = tempname()
    return path, UInt8.(rand_int(SplitMix(4), 2^16, 0, 255))
end
function file_roundtrip(path, chunk)
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
    rm(path; force = true)
    return n
end
bench(file_roundtrip, file_roundtrip_setup; ops = 64, tolerance = 0.15)


# Many small reads from a file: per-call overhead of read! and readbytes! on an IOStream
# (#62856, #63788)
function small_reads_setup()
    path = tempname()
    write(path, UInt8.(rand_int(SplitMix(42), 2^20, 0, 255)))
    io = open(path)
    # The open stream keeps the data; Windows cannot remove an open file
    Sys.iswindows() || rm(path)
    return io, zeros(4), zeros(UInt8, 32)
end
function small_reads(io, a, b)
    seekstart(io)
    s = 0
    for _ in 1:2^13
        read!(io, a)
        readbytes!(io, b, 32)
        s += b[1]
    end
    return s
end
bench(small_reads, small_reads_setup; ops = 2 * 2^13)

end
