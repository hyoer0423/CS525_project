from general_fusion_pipeline import GeneralFusionPipeline
from hloenv import AltPipeline, HloEnv, HloPass, Pass, Pipeline
import os,pathlib

algebraic_config_options = { "replace_transpose_with_bitcast": False}

pass_list = [
    HloPass.AllToAllDecomposer(),
    HloPass.OperandUpcaster(),
    HloPass.ResultCaster(),
    HloPass.RngExpander(),
    HloPass.ComparisonExpander(),
    HloPass.ZeroSizedHloElimination(),
    HloPass.GpuScatterExpander(),
    HloPass.QrExpander(),
    HloPass.EighExpander(),
    HloPass.DynamicIndexSplitter(),
    HloPass.CallInliner(),
    HloPass.DotDecomposer(),
    HloPass.Convolution4DExpander(),
    HloPass.StableSortExpander(),
    # HloPass.BFloat16Normalization(True),
    HloPass.BatchNormExpander(True, True, True),
    HloPass.ConditionalCanonicalizer(),
    HloPass.DynamicDimensionSimplifier(),
    HloPass.AlgebraicSimplifier(options=algebraic_config_options),
    HloPass.BitcastDtypesExpander(),
    HloPass.SortSimplifier(),
    HloPass.TupleSimplifier(),
    HloPass.WhileLoopConstantSinking(),
    HloPass.WhileLoopSimplifier(),
    HloPass.ReshapeMover(), 
    HloPass.HloConstantFolding(),
    HloPass.ConditionalSimplifier(),
    HloPass.RealImagExpander(),
    HloPass.TransposeFolding(),
    HloPass.HloCSE(is_layout_sensitive=False),
    HloPass.HloDCE(),
    HloPass.WhileLoopTripCountAnnotator(),
    HloPass.AllReduceFolder(),
    HloPass.ReduceScatterCreator(),
    HloPass.AllReduceReassociate(),
    HloPass.AllGatherBroadcastReorder(),
    HloPass.GpusolverRewriter(),
    HloPass.GpuConvRewriter(),
    HloPass.CudnnFusedConvRewriter(),
    HloPass.GpuConvPaddingLegalization(),
    # HloPass.CudnnPadForConvolutions(),
    # HloPass.CudnnVectorizeConvolutions(),
    HloPass.FlattenCallGraph(),
    HloPass.HloConstantFolding(),
    HloPass.ReductionDegenerateDimRemover(),
    HloPass.ReductionLayoutNormalizer(),
    HloPass.ReductionDimensionGrouper(),
    HloPass.ReductionSplitter(),
    # HloPass.GpuTreeReductionRewriter(),
    HloPass.TransposeFolding(),
    HloPass.GemmRewriter(),
    HloPass.GemmBroadcastFoldingRewriter(),
    # HloPass.BFloat16Normalization(False),
    # HloPass.GpuConvAlgorithmPicker(),
    HloPass.HloCSE(True),
    # HloPass.GemmAlgorithmPicker(),
    HloPass.TriangularSolveRewriter(),
    HloPass.VariadicOpSplitter(),
    # HloPass.GpuInstructionFusion(False),
    # HloPass.GpuInstructionFusion(True),
    HloPass.FusionMerger(),
    # HloPass.GpuMultiOutputFusion(),
    HloPass.HloCSE(True, True),
    # HloPass.GpuHorizontalLoopFusion(),
    # HloPass.GpuHorizontalInputFusion(),
    HloPass.AllReduceContiguous(),
    HloPass.AsyncCollectiveCreator(),
    HloPass.CollectivesScheduleLinearizer(),
    HloPass.OptimizationBarrierExpander(),


]

def hlo_generator():
    directory_path = "/root/hloenv/examples/hlo_datasets/hlos"
    cnt = 0
    for dirpath, dirnames, filenames in os.walk(directory_path):
        for filename in filenames:
            relative_file_path = os.path.join(dirpath, filename)
            absolute_file_path = os.path.abspath(relative_file_path)
            yield absolute_file_path
            cnt += 1
            if cnt == 20:
                return 
          
        



def count_conflict(pass1, pass2):
    # path_list = [
    #     "hlo_texts/jax-md/module_0013.jit__lambda_.7.before_optimizations.txt",
    #     "hlo_texts/jax-md/module_0029.jit_safe_mask.15.before_optimizations.txt",
    #     "hlo_texts/jax-md/module_0190.jit_safe_mask.61.before_optimizations.txt",
    # ]
    total_conflict = 0
    for hlo_graph in hlo_generator():
        hlo_path = hlo_graph
        hlo_env = HloEnv(hlo_path, "gpu")
        pipeline = Pipeline("test")

        pipeline.add_pass(pass1)
        pipeline.add_pass(pass2)

        dry_pipeline = AltPipeline(pipeline)

        hlo_env.run(dry_pipeline)

        hlo_graph = hlo_env.get_hlo_graph(do_hash_verification=False)
        total_conflict += len(hlo_graph.alternative_indices)

    return total_conflict



def count_conflict_for_all_passes():
    conflict_list = []
    with open("/root/hloenv/examples/conflict_passes_result.text", "w") as output:
        for pass1 in pass_list:
            for pass2 in pass_list:
                if pass1 != pass2:
                    try:
                        total_conflict = count_conflict(Pass(pass1), Pass(pass2))
                    except Exception as e:
                        print("Ignore:", e)
                    pass1_name = str(pass1).replace("<hloenv.python.py_hlo_env.hlo_pass.","").split(" ")[0]
                    pass2_name = str(pass2).replace("<hloenv.python.py_hlo_env.hlo_pass.","").split(" ")[0]      
                            
                    output.write(pass1_name + " " + pass2_name + " " + str(total_conflict)+ "\n")
                    if total_conflict != 0:
                        print("conflict number:", pass1_name, pass2_name, total_conflict)
                    else:
                        print("No conflict:", pass1_name, pass2_name, total_conflict)
                    
                
                  
                    
count_conflict_for_all_passes()
# python3 ./lyc.py 2>&1 | grep "conflict number:"
# python3 ./lyc.py 2>&1 | grep "No conflict:"