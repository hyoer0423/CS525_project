from general_fusion_pipeline import GeneralFusionPipeline
from hloenv import AltPipeline, HloEnv, HloPass, Pass, Pipeline
import os,pathlib
import sys
import numpy as np
args=sys.argv[1:]
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


          
        



def count_conflict(pass1, pass2):
  
    hlo_path = os.path.join( pathlib.Path(__file__).parent.absolute(),"hlo_texts/jax-md/module_0013.jit__lambda_.7.before_optimizations.txt")
   
    
    hlo_env = HloEnv(hlo_path, "gpu")
    pipeline = Pipeline("test")
    try:
        print(str(pass1))
        pipeline.add_pass(pass1)
    except:
        print('')
    try:
        print(str(pass2))
        pipeline.add_pass(pass2)
    except:
        print('')

    hlo_env.run(pipeline)
    return hlo_env

def draw(hlo_env):
    hlo_graph = hlo_env.get_hlo_graph()
    print("=========graph_features==========")
    print(hlo_graph.out_edge_offsets)
    print(len(hlo_graph.out_edge_offsets))
    print(hlo_graph.out_edge_indices)
    print(len(hlo_graph.out_edge_indices))
    print(hlo_graph.in_edge_offsets)
    print(hlo_graph.in_edge_indices)
    print(hlo_graph.alternative_indices)
    print(hlo_graph.opcode_attr_counts)
    print(hlo_graph.hash())

    node_features = hlo_graph.node_features
    in_edge_features = hlo_graph.in_edge_features
    out_edge_features = hlo_graph.out_edge_features

    print("=========node_features==========")
    print(node_features.uids)
    print(node_features.gids)
    print(node_features.num_users)
    print(node_features.num_operands)
    print(node_features.opcodes)
    print(node_features.opcode_attrs)
    print(node_features.num_opcode_attrs.reshape(-1, 2))
    print(node_features.is_alternative)
    print(node_features.in_tensor_sizes)
    print(node_features.out_tensor_sizes)
    print(node_features.has_max_in_tensor)
    print(node_features.has_max_out_tensor)
    print(node_features.names)

    print("=========in_edge_features===========")
    print(in_edge_features.uids)
    print(in_edge_features.srcs)
    print(in_edge_features.dsts)
    print(in_edge_features.dims.reshape(-1, 8))
    print(in_edge_features.layouts.reshape(-1, 8))
    print(in_edge_features.dtypes)

    print("=========out_edge_features===========")
    print(out_edge_features.uids)
    print(out_edge_features.srcs)
    print(out_edge_features.dsts)
    print(np.array(out_edge_features.dims).reshape(-1, 8))
    print(np.array(out_edge_features.layouts).reshape(-1, 8))
    print(out_edge_features.dtypes)



pass1=pass_list[int(args[0])]
if len(args)>1:
    pass2=pass_list[int(args[1])]
    hlo_env=count_conflict(pass1, pass2)
else:
    hlo_env=count_conflict(pass1)
draw(hlo_env)
                
                  