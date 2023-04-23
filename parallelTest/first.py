from general_fusion_pipeline import GeneralFusionPipeline
from hloenv import AltPipeline, HloEnv, HloPass, Pass, Pipeline
import os,pathlib
import time

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

def testTime(pass_this, hlo_path):
    
    start = time.time()
    
    hlo_env = HloEnv(hlo_path, "gpu")
    pipeline = Pipeline("test")
    pipeline.add_pass(pass_this)
    dry_pipeline = AltPipeline(pipeline)
    hlo_env.run(dry_pipeline)
    hlo_graph = hlo_env.get_hlo_graph(do_hash_verification=False)
    
    end = time.time()
    
    print(end - start)
    