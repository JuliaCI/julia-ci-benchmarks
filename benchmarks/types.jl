# The type system at run time (src/subtype.c, jltypes.c, gf.c): subtyping, intersection,
# type construction through the type cache, method lookup and type printing. Inference
# and dispatch spend much of their time here.
module TypesBench
using ..Driver

# Shared by the benchmarks below: paste this part first to run one of them in the REPL.
const TYPES = Any[
    Int, Float64, Real, Number, Any, String, AbstractString, Union{Int,Nothing},
    Vector{Int}, Vector{<:Real}, AbstractVector{Float64}, Array{T,2} where T,
    Tuple{Int,Float64}, Tuple{Vararg{Int}}, Tuple{Int,Vararg{Real}}, NTuple{4,Float64},
    Dict{String,Any}, AbstractDict{K,V} where {K<:AbstractString,V}, Pair{Symbol,Int},
    Type{Int}, Type{<:Number}, Base.RefValue{Any}, Ref{T} where T<:Integer,
    Union{Missing,Float64,String}, Tuple{T,T} where T<:Number, Tuple{Vector{T},T} where T,
    Base.Generator{Vector{Int},typeof(identity)}, NamedTuple{(:a, :b),Tuple{Int,String}},
    Val{3}, Function, typeof(sin), Matrix{Union{Nothing,Int}},
]

function pairs_input(seed)
    r = SplitMix(seed)
    n = length(TYPES)
    return [(TYPES[i], TYPES[j]) for (i, j) in zip(rand_int(r, 4096, 1, n), rand_int(r, 4096, 1, n))]
end
# End of the shared part.

subtype_setup() = pairs_input(1)
function subtype(ps)
    n = 0
    for (a, b) in ps
        n += (a <: b) + (b <: a)
    end
    return n
end
bench(subtype, subtype_setup; ops = 4096)

typeintersect_pairs_setup() = pairs_input(2)
function typeintersect_pairs(ps)
    n = 0
    for (a, b) in ps
        n += typeintersect(a, b) === Union{}
    end
    return n
end
bench(typeintersect_pairs, typeintersect_pairs_setup; ops = 4096)

# apply_type with cache hits: the parameters vary, the types already exist
function apply_type_setup()
    r = SplitMix(3)
    elts = Any[Int, Float64, String, Symbol, Nothing, Any, Int8, ComplexF64]
    return [(elts[i], j) for (i, j) in zip(rand_int(r, 4096, 1, 8), rand_int(r, 4096, 1, 6))]
end
function apply_type(ps)
    n = 0
    for (T, k) in ps
        n += isconcretetype(Vector{T}) + isconcretetype(NTuple{k,T}) + isconcretetype(Dict{Symbol,T}) +
             isconcretetype(Tuple{T,Vararg{Int,k}})
    end
    return n
end
bench(apply_type, apply_type_setup; ops = 4096)

# Method table matching (jl_matching_methods), as reflection and inference do it
function method_lookup_setup()
    r = SplitMix(4)
    fs = Any[+, *, getindex, convert, show, length, iterate, ==]
    args = Any[Int, Float64, Vector{Int}, String, Any, Missing, Dict{Int,Int}, Tuple{Int,Int}]
    return [(fs[i], Tuple{args[j], args[k]}) for (i, j, k) in zip(rand_int(r, 1536, 1, 8), rand_int(r, 1536, 1, 8), rand_int(r, 1536, 1, 8))]
end
function method_lookup(qs)
    n = 0
    for (f, tt) in qs
        ms = Base._methods_by_ftype(Tuple{typeof(f), tt.parameters...}, 4, Base.get_world_counter())
        # Past the limit of 4 matches the lookup gives up and returns `nothing` (or `false`)
        n += hasmethod(f, tt) + (ms isa Vector ? length(ms) : 5)
    end
    return n
end
bench(method_lookup, method_lookup_setup; ops = 1536)

# Type printing, alias search included: error messages, the REPL, stack traces
show_types_setup() = TYPES
function show_types(ts)
    n = 0
    for T in ts
        n += sizeof(string(T)) + sizeof(sprint(show, Vector{T}))
    end
    return n
end
bench(show_types, show_types_setup; ops = length(TYPES))

end
