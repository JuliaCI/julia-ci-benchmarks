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
    return io
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

# show of containers into an IOBuffer: text/plain display of a matrix (column alignment),
# a Vector{Float64} under :compact, a Dict and a NamedTuple
function show_containers_setup()
    r = SplitMix(70)
    M = reshape(rand_f64(r, 24 * 8, -100, 100), 24, 8)
    v = rand_f64(r, 200)
    d = Dict(string("key", i) => i / 3 for i in 1:64)
    nt = (a = 1, b = "two", c = [3.0, 4.0], d = (5, :six))
    return M, v, d, nt
end
function show_containers(M, v, d, nt)
    io = IOBuffer()
    show(io, MIME"text/plain"(), M)
    show(IOContext(io, :compact => true), v)
    show(io, d)
    for _ in 1:16
        show(io, nt)
    end
    return String(take!(io))
end
bench(show_containers, show_containers_setup; ops = 24 * 8 + 200 + 64 + 16)

# Bit types written to and read back from an IOBuffer: write(io, x) of each value, buffer
# growth, and read(io, T)
binary_rw_setup() = (rand_int(SplitMix(71), 2^13, -10^15, 10^15), rand_f64(SplitMix(72), 2^13))
function binary_rw(is, fs)
    io = IOBuffer()
    for i in eachindex(is, fs)
        write(io, is[i])
        write(io, fs[i])
        write(io, Int32(i))
        write(io, UInt8(i & 0xff))
    end
    seekstart(io)
    s = 0
    t = 0.0
    while !eof(io)
        s += read(io, Int)
        t += read(io, Float64)
        s += read(io, Int32) + read(io, UInt8)
    end
    return io, s, t
end
bench(binary_rw, binary_rw_setup; ops = 8 * 2^13)

# A libuv pipe within the process: a writer task hands 1 MiB to the reading task in
# 16 KiB writes, through uv_write, the event loop and task switches. Wall time, since
# the event loop can wait in the kernel.
pipe_roundtrip_setup() = (UInt8.(rand_int(SplitMix(73), 2^14, 0, 255)), zeros(UInt8, 2^20))
function pipe_roundtrip(chunk, buf)
    p = Pipe()
    Base.link_pipe!(p; reader_supports_async = true, writer_supports_async = true)
    w = @async begin
        for _ in 1:64
            write(p.in, chunk)
        end
        close(p.in)
    end
    read!(p.out, buf)
    wait(w)
    close(p.out)
    return buf
end
bench(pipe_roundtrip, pipe_roundtrip_setup; ops = 64, metric = "time", tolerance = 0.15)

end
