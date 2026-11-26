# Julia-level compilation of fixed call graphs: abstract interpretation alone,
# optimization alone, and both, after BaseBenchmarks' "inference" group. An
# AbstractInterpreter with its own code cache infers the whole call graph on every sample.
# What it infers is data/inference_workload.jl, a program that calls no function of Base,
# so that a result moves with the compiler and not with a change to a method of Base that
# a benchmark happened to reach. The interpreter API differs between julia versions; the
# gates below follow BaseBenchmarks, and a break only fails this category.
module InferenceBench
using ..Driver

# Shared by the benchmarks below: paste this part first to run one of them in the REPL.
@static if VERSION ≥ v"1.12.0-DEV.1581"
    const CC = Base.REFLECTION_COMPILER[] === nothing ? Base.Compiler : Base.REFLECTION_COMPILER[]
else
    const CC = Core.Compiler
end
using Core: MethodInstance, CodeInstance, SimpleVector
using .CC: AbstractInterpreter, InferenceParams, InferenceResult, InferenceState, OptimizationParams,
           OptimizationState, WorldRange, specialize_method
@static if isdefined(CC, :WorldView)
    import .CC: WorldView
end
@static if VERSION ≥ v"1.11.0-DEV.1498"
    import .CC: get_inference_world
else
    import .CC: get_world_counter as get_inference_world
end
@static if VERSION ≥ v"1.14.0-DEV.1691"
    const InfCache = CC.InferenceCache
    empty_inf_cache() = CC.InferenceCache()
else
    const InfCache = Vector{InferenceResult}
    empty_inf_cache() = InferenceResult[]
end

struct BenchCache
    dict::IdDict{MethodInstance,CodeInstance}
end
struct BenchInterpreter <: AbstractInterpreter
    world::UInt
    inf_params::InferenceParams
    opt_params::OptimizationParams
    optimize::Bool
    inf_cache::InfCache
    code_cache::BenchCache
end
BenchInterpreter(; optimize::Bool = true) =
    BenchInterpreter(Base.get_world_counter(), InferenceParams(), OptimizationParams(), optimize,
                     empty_inf_cache(), BenchCache(IdDict{MethodInstance,CodeInstance}()))
CC.may_optimize(interp::BenchInterpreter) = interp.optimize
CC.may_compress(::BenchInterpreter) = true
CC.may_discard_trees(::BenchInterpreter) = true
CC.InferenceParams(interp::BenchInterpreter) = interp.inf_params
CC.OptimizationParams(interp::BenchInterpreter) = interp.opt_params
get_inference_world(interp::BenchInterpreter) = interp.world
CC.get_inference_cache(interp::BenchInterpreter) = interp.inf_cache
@static if isdefined(CC, :WorldView)
    CC.code_cache(interp::BenchInterpreter) = WorldView(interp.code_cache, WorldRange(get_inference_world(interp)))
    CC.get(wvc::WorldView{BenchCache}, mi::MethodInstance, default) = get(wvc.cache.dict, mi, default)
    CC.getindex(wvc::WorldView{BenchCache}, mi::MethodInstance) = getindex(wvc.cache.dict, mi)
    CC.haskey(wvc::WorldView{BenchCache}, mi::MethodInstance) = haskey(wvc.cache.dict, mi)
    CC.setindex!(wvc::WorldView{BenchCache}, ci::CodeInstance, mi::MethodInstance) = setindex!(wvc.cache.dict, ci, mi)
else
    CC.code_cache(interp::BenchInterpreter) = CC.OverlayCodeCache(interp.code_cache, interp.inf_cache)
    CC.get(cache::BenchCache, mi::MethodInstance, default) = get(cache.dict, mi, default)
    CC.getindex(cache::BenchCache, mi::MethodInstance) = getindex(cache.dict, mi)
    CC.haskey(cache::BenchCache, mi::MethodInstance) = haskey(cache.dict, mi)
    CC.setindex!(cache::BenchCache, ci::CodeInstance, mi::MethodInstance) = setindex!(cache.dict, ci, mi)
end
@static if isdefined(CC, :cache_owner)
    CC.cache_owner(interp::BenchInterpreter) = interp.code_cache
end

# Infer f(::types...) with a fresh interpreter, so nothing is cached from before
function infer(@nospecialize(f), @nospecialize(types::Tuple); optimize::Bool = true, run_optimizer::Bool = optimize)
    interp = BenchInterpreter(; optimize)
    match = Base._which(Tuple{Core.Typeof(f), types...}; world = get_inference_world(interp))
    mi = specialize_method(match.method, match.spec_types, match.sparams)::MethodInstance
    frame = InferenceState(InferenceResult(mi), run_optimizer ? :global : :no, interp)::InferenceState
    CC.typeinf(interp, frame)
    frame.bestguess !== Union{} || error("inference found no return type")
    return frame, interp
end
# These kernels allocate millions of objects, and their cycles vary 2-11% between processes
# with the heap, even when each sample starts after a full collection (the untimed `reset`,
# which keeps the instruction count within 0.5%). So they are judged on instructions:
# compiler work moved at most 0.2% in instructions between builds that differ only in code
# placement.
# Each sample optimizes a frame that `reset` inferred without optimizing
optimize!(state) = (CC.optimize(state[][2], OptimizationState(state[]...), state[][1].result); state[][1])
infer_unoptimized!(state, f, types) = (state[] = infer(f, types; run_optimizer = false); GC.gc(); nothing)

# The program to infer
include(joinpath(Driver.DATA_DIR, "inference_workload.jl"))
const W = InferenceWorkload
# What a kernel infers must be the workload's own (closures and exceptions also reach
# constructors in Core): run on every kernel once, when it is defined
function check_frozen(kernel)
    _, interp = kernel()
    for mi in keys(interp.code_cache.dict)
        m = mi.def
        m isa Method && (m.module === W || m.module === Core) ||
            error("$(nameof(kernel)) infers $mi, which is not part of the workload")
    end
    return kernel
end

# Generated functions that stress one part of the compiler each (from BaseBenchmarks, with
# the julia PRs that motivated them), on the workload's functions
# Many locals: sparse state management (julia#45276)
@eval W function many_local_vars(x)
    $(let ex = Expr(:block), var = gensym()
        push!(ex.args, :($var = x))
        for _ in 1:10000
            v = gensym()
            push!(ex.args, :($v = $var + 1))
            var = v
        end
        ex
    end)
end
# Many method matches of the same signature: CachedMethodTable (julia#46535)
@eval W function many_method_matches(x)
    y = sum(x)
    $([:(fill!(x, $(Float64(i))); y += sum(x)) for i in 1:100]...)
    return y
end
# Concrete evaluation of a chain of 1000 constant calls
@eval W sins(x) = $(let ex = Expr(:block), var = :x
        for _ in 1:1000
            v = gensym()
            push!(ex.args, :($v = sin($var)))
            var = v
        end
        ex
    end)
@eval W many_const_calls() = sins(42)
@eval W function many_invoke_calls(x)
    y = sum(x)
    $([:(y += invoke(strangesum, Tuple{Obj}, x)) for _ in 1:100]...)
    return y
end
@eval W function many_opaque_closures(x)
    y = sum(x)
    $([:(oc = Base.Experimental.@opaque (i, x, y) -> (fill!(x, convert(Float64, i)); y += sum(x)); oc($i, x, y)) for i in 1:100]...)
    return y
end
# Many entries in the const-prop cache for one method: a callee that can't be concretely
# evaluated, called with many distinct constants (julia#64707)
@eval W function many_constprop_entries()
    y = 0.0
    $([:(y += constprop_target($(Float64(i)))) for i in 1:1000]...)
    return y
end
# End of the shared part.

# Abstract interpretation alone (no optimization, so no concrete evaluation either) of the
# workload's kernels: numbers with promotion, printing of values of any type, and a lazy
# broadcast
abstract_numeric() = infer(W.numeric_kernel, (Int,); optimize = false)
bench(check_frozen(abstract_numeric); seconds = 0.25, metric = "instructions", reset = GC.gc)

abstract_show() = infer(W.show_kernel, (W.Buf, W.Quote); optimize = false)
bench(check_frozen(abstract_show); seconds = 0.25, metric = "instructions", reset = GC.gc)

abstract_show_collections() = infer(W.show_collections_kernel, (W.Buf, W.Tree{Tuple{Int,W.Rat{Int}},W.List{W.Cplx{Float64}}}, W.Vec{W.Pair{Symbol,W.Dual{Float64}}}); optimize = false)
bench(check_frozen(abstract_show_collections); seconds = 0.25, metric = "instructions", reset = GC.gc)

abstract_broadcast() = infer(W.broadcast_kernel, (W.Vec{Float64}, W.Vec{W.Rat{Int}}, Float64); optimize = false)
bench(check_frozen(abstract_broadcast); seconds = 0.25, metric = "instructions", reset = GC.gc)

# Inference with optimization of every frame, as the compiler runs it
infer_rand() = infer(W.rand_kernel, (W.Xoshiro,))
bench(check_frozen(infer_rand); seconds = 0.25, metric = "instructions", reset = GC.gc)

# Orderings built from functions, whose type depends on a value, in a recursive sort
infer_sort() = infer(W.sort_kernel, (W.List{Float64}, W.List{W.Pair{Int,W.Rat{Int}}}, Bool))
bench(check_frozen(infer_sort); seconds = 0.25, metric = "instructions", reset = GC.gc)

infer_many_local_vars() = infer(W.many_local_vars, (Int,))
bench(check_frozen(infer_many_local_vars); seconds = 0.25, metric = "instructions", reset = GC.gc)

infer_many_method_matches() = infer(W.many_method_matches, (W.Vec{Float64},))
bench(check_frozen(infer_many_method_matches); seconds = 0.25, metric = "instructions", reset = GC.gc)

infer_many_const_calls() = infer(W.many_const_calls, ())
bench(check_frozen(infer_many_const_calls); seconds = 0.25, metric = "instructions", reset = GC.gc)

infer_many_constprop_entries() = infer(W.many_constprop_entries, ())
bench(check_frozen(infer_many_constprop_entries); seconds = 0.25, metric = "instructions", reset = GC.gc)

infer_many_invoke_calls() = infer(W.many_invoke_calls, (W.Vec{Float64},))
bench(check_frozen(infer_many_invoke_calls); seconds = 0.25, metric = "instructions", reset = GC.gc)

infer_many_opaque_closures() = infer(W.many_opaque_closures, (W.Vec{Float64},))
bench(check_frozen(infer_many_opaque_closures); seconds = 0.25, metric = "instructions", reset = GC.gc)

# The optimizer alone on one frame, inferred without optimization in the untimed reset
optimize_many_local_vars_setup() = Ref{Any}()
optimize_many_local_vars(state) = optimize!(state)
bench(optimize_many_local_vars, optimize_many_local_vars_setup; seconds = 0.25, metric = "instructions",
      reset = state -> infer_unoptimized!(state, W.many_local_vars, (Int,)))

optimize_many_method_matches_setup() = Ref{Any}()
optimize_many_method_matches(state) = optimize!(state)
bench(optimize_many_method_matches, optimize_many_method_matches_setup; seconds = 0.25, metric = "instructions",
      reset = state -> infer_unoptimized!(state, W.many_method_matches, (W.Vec{Float64},)))

optimize_many_invoke_calls_setup() = Ref{Any}()
optimize_many_invoke_calls(state) = optimize!(state)
bench(optimize_many_invoke_calls, optimize_many_invoke_calls_setup; seconds = 0.25, metric = "instructions",
      reset = state -> infer_unoptimized!(state, W.many_invoke_calls, (W.Vec{Float64},)))

optimize_many_opaque_closures_setup() = Ref{Any}()
optimize_many_opaque_closures(state) = optimize!(state)
bench(optimize_many_opaque_closures, optimize_many_opaque_closures_setup; seconds = 0.25, metric = "instructions",
      reset = state -> infer_unoptimized!(state, W.many_opaque_closures, (W.Vec{Float64},)))

end
