import torch, ttnn, atexit, time
from torch import nn
from typing import Tuple, Callable, Dict
from collections import defaultdict
from functools import wraps
from models.utility_functions import is_wormhole_b0, is_blackhole
from loguru import logger

"""
Module Implementation Map:

PyTorch-Only Modules (nn.Module):
├── InputEmbedderModule
├── DiffusionConditioningModule
├── ConfidenceModule
└── AffinityModule

Hybrid Modules (TorchWrapper - supports both PyTorch & TTNN):
├── MSAModule
│   └── MSA (TTNN)
│       ├── MSALayer
│       └── OuterProductMean
├── PairformerModule
│   └── Pairformer (TTNN)
│       ├── PairformerLayer
│       ├── TriangleMultiplication
│       └── TriangleAttention
└── DiffusionTransformerModule
    └── DiffusionTransformer (TTNN)
        ├── DiffusionTransformerLayer
        ├── AttentionPairBias
        ├── AdaLN
        └── ConditionedTransitionBlock

TTNN-Only Base Classes:
├── Module (Base TTNN class)
└── TorchWrapper (Base hybrid PyTorch/TTNN class)
"""

TRIANGLE_MULT_CHUNK_SIZE = 32
TRANSITION_CHUNK_SIZE = 64
USE_FLOAT32 = False

device = None
timing_summary_printed = False

# Statistics storage for timing instrumentation
timing_stats = defaultdict(lambda: {"total_time": 0.0, "calls": 0, "is_top_level": False})
conversion_stats = {
    "torch_to_ttnn": {"total_time": 0.0, "calls": 0, "modules": defaultdict(lambda: {"total_time": 0.0, "calls": 0})},
    "ttnn_to_torch": {"total_time": 0.0, "calls": 0, "modules": defaultdict(lambda: {"total_time": 0.0, "calls": 0})},
    "ttnn_ops": {"total_time": 0.0, "calls": 0},
    "torch_ops": {"total_time": 0.0, "calls": 0},
}

# Mark these as top-level modules
TOP_LEVEL_MODULES = {
    "InputEmbedderModule",
    "MSAModule", 
    "PairformerModule",
    "DiffusionConditioningModule",
    "AtomDiffusionModule",
    "ConfidenceModule",
    "AffinityModule",
}

def timing_decorator(func, is_torch=False):
    """Decorator to measure timing of operations
    Args:
        is_torch: Whether this is a PyTorch operation (vs ttnn)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        
        # Update timing statistics
        func_name = f"{func.__qualname__}()"
        if func_name not in timing_stats:
            # Determine module type:
            class_name = func.__qualname__.split('.')[0]
            
            # 1. TTNN modules inherit from Module and use ttnn.Tensor
            is_ttnn_module = (
                not is_torch and  # Not explicitly marked as PyTorch
                (
                    # Direct TTNN modules in tenstorrent.py
                    class_name in [
                        "MSA", "MSALayer", "PairWeightedAveraging",
                        "Pairformer", "PairformerLayer", "AttentionPairBias", "TriangleMultiplication",
                        "DiffusionTransformer", "DiffusionTransformerLayer", "AdaLN", "ConditionedTransitionBlock",
                        "AtomTransformer", "AtomAttentionEncoder", "AtomAttentionDecoder"
                    ] or
                    # Any class that inherits from Module (not TorchWrapper)
                    (hasattr(args[0], '__class__') and issubclass(args[0].__class__, Module) and not issubclass(args[0].__class__, TorchWrapper))
                )
            )
            
            # 2. Hybrid modules are TorchWrapper classes that provide PyTorch interface to TTNN
            is_hybrid_module = (
                class_name in [
                    "MSAModule", "PairformerModule", "DiffusionTransformerModule"
                ] or
                (hasattr(args[0], '__class__') and issubclass(args[0].__class__, TorchWrapper))
            )
            
            timing_stats[func_name] = {
                "total_time": 0,
                "calls": 0,
                "is_top_level": False,
                "is_ttnn": is_ttnn_module,
                "is_hybrid": is_hybrid_module
            }
            
        timing_stats[func_name]["total_time"] += elapsed_time
        timing_stats[func_name]["calls"] += 1
        
        # Mark as top-level if the class name is in TOP_LEVEL_MODULES
        class_name = func.__qualname__.split('.')[0]
        timing_stats[func_name]["is_top_level"] = class_name in TOP_LEVEL_MODULES
        
        # Track operation type and conversions
        if timing_stats[func_name]["is_ttnn"]:
            conversion_stats["ttnn_ops"]["total_time"] += elapsed_time
            conversion_stats["ttnn_ops"]["calls"] += 1
        elif timing_stats[func_name]["is_hybrid"]:
            # Track both TTNN and conversion time for hybrid modules
            conversion_stats["ttnn_ops"]["total_time"] += elapsed_time
            conversion_stats["ttnn_ops"]["calls"] += 1
            
            # Track conversion time for this module
            if "torch_to_ttnn" in func_name or "ttnn_to_torch" in func_name:
                op = "torch_to_ttnn" if "torch_to_ttnn" in func_name else "ttnn_to_torch"
                module_name = func.__qualname__.split('.')[0]
                conversion_stats[op]["modules"][module_name]["total_time"] += elapsed_time
                conversion_stats[op]["modules"][module_name]["calls"] += 1
        else:
            conversion_stats["torch_ops"]["total_time"] += elapsed_time
            conversion_stats["torch_ops"]["calls"] += 1
        
        return result
    return wrapper

def torch_timing_decorator(func):
    """Decorator to measure timing of PyTorch operations"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        
        # Update timing statistics
        func_name = f"{func.__qualname__}()"
        if func_name not in timing_stats:
            timing_stats[func_name] = {
                "total_time": 0,
                "calls": 0,
                "is_top_level": False,
                "is_ttnn": False  # This is a PyTorch operation
            }
            
        timing_stats[func_name]["total_time"] += elapsed_time
        timing_stats[func_name]["calls"] += 1
        
        # Track as torch operation if not a wrapper
        module_name = func.__qualname__.split('.')[0]
        if not any(ttnn_name in module_name for ttnn_name in ["Pairformer", "MSA", "DiffusionTransformer", "TorchWrapper"]):
            conversion_stats["torch_ops"]["total_time"] += elapsed_time
            conversion_stats["torch_ops"]["calls"] += 1
        
        return result
    return wrapper

def print_timing_summary():
    """Print hierarchical timing summary comparing PyTorch vs TTNN operations"""
    global timing_summary_printed
    if timing_summary_printed:
        return
    
    if not timing_stats and not any(stats["calls"] > 0 for stats in conversion_stats.values()):
        return
    
    timing_summary_printed = True

    # Build hierarchy with proper base structure and visual formatting
    hierarchy = {}
    
    # Helper to initialize a module in hierarchy
    def init_module(name, module_type="unknown"):
        if name not in hierarchy:
            hierarchy[name] = {
                "total_time": 0,
                "calls": 0,
                "type": module_type,
                "children": set(),
                "parent": None,
                "methods": set(),
                "direct_timing": False  # Track if module has direct timing data
            }
        return hierarchy[name]
    
    # Manually create the base hierarchy structure first
    base_hierarchy = {
        "Boltz2": {
            "children": [
                "InputEmbedder",
                "MSAModule",
                "PairformerModule",
                "DistogramModule",
                "BFactorModule",
                "ContactConditioning",
                "TemplateModule",
                "TemplateV2Module",
                "AtomDiffusion",
                "DiffusionConditioning",
                "ConfidenceModule",
                "AffinityModule",
                "FourierEmbedding"
            ],
            "type": "pytorch"  # Main model is PyTorch
        },
        # Input processing
        "InputEmbedder": {
            "children": ["AtomEncoder", "AtomAttentionEncoder", "RelativePositionEncoder"],
            "type": "pytorch"  # Regular PyTorch module
        },
        "AtomEncoder": {
            "children": [],
            "type": "pytorch"  # Regular PyTorch module
        },
        "RelativePositionEncoder": {
            "children": [],
            "type": "pytorch"  # Regular PyTorch module
        },
        # MSA processing
        "MSAModule": {
            "children": ["MSA"],
            "type": "ttnn"  # TTNN module with PyTorch interface
        },
        "MSA": {
            "children": ["MSALayer", "PairWeightedAveraging"],
            "type": "ttnn"  # TTNN implementation
        },
        "MSALayer": {
            "children": [],
            "type": "ttnn"  # TTNN implementation
        },
        "PairWeightedAveraging": {
            "children": [],
            "type": "ttnn"  # TTNN implementation
        },
        # Pairwise features
        "PairformerModule": {
            "children": ["Pairformer"],
            "type": "ttnn"  # TTNN module with PyTorch interface
        },
        "Pairformer": {
            "children": ["PairformerLayer", "AttentionPairBias", "TriangleMultiplication"],
            "type": "ttnn"  # TTNN implementation
        },
        "PairformerLayer": {
            "children": [],
            "type": "ttnn"  # TTNN implementation
        },
        "AttentionPairBias": {
            "children": [],
            "type": "ttnn"  # TTNN implementation
        },
        "TriangleMultiplication": {
            "children": [],
            "type": "ttnn"  # TTNN implementation
        },
        # Structure prediction
        "AtomDiffusion": {
            "children": ["DiffusionModule"],
            "type": "pytorch"  # Regular PyTorch module
        },
        "DiffusionModule": {
            "children": ["AtomAttentionEncoder", "DiffusionTransformerModule", "AtomAttentionDecoder", "SingleConditioning"],
            "type": "pytorch"  # Regular PyTorch module that uses TTNN components
        },
        "DiffusionTransformerModule": {
            "children": ["DiffusionTransformer"],
            "type": "ttnn"  # TTNN module with PyTorch interface
        },
        "DiffusionTransformer": {
            "children": ["DiffusionTransformerLayer", "AdaLN", "ConditionedTransitionBlock"],
            "type": "ttnn"  # TTNN implementation of transformer
        },
        "DiffusionTransformerLayer": {
            "children": [],
            "type": "ttnn"  # TTNN implementation
        },
        "AdaLN": {
            "children": [],
            "type": "ttnn"  # TTNN implementation
        },
        "ConditionedTransitionBlock": {
            "children": [],
            "type": "ttnn"  # TTNN implementation
        },
        "AtomAttentionEncoder": {
            "children": ["AtomTransformer"],
            "type": "pytorch"  # PyTorch module that uses TTNN
        },
        "AtomAttentionDecoder": {
            "children": ["AtomTransformer"],
            "type": "pytorch"  # PyTorch module that uses TTNN
        },
        "AtomTransformer": {
            "children": ["DiffusionTransformerModule"],
            "type": "pytorch"  # PyTorch module that uses TTNN transformer
        },
        # Confidence prediction
        "ConfidenceModule": {
            "children": ["PairformerModule", "ConfidenceHeads"],
            "type": "pytorch"  # Regular PyTorch module that uses TTNN components
        },
        "ConfidenceHeads": {
            "children": [],
            "type": "pytorch"  # Regular PyTorch module with prediction heads
        },
        # Additional prediction heads
        "DistogramModule": {
            "children": [],
            "type": "pytorch"  # Regular PyTorch module
        },
        "BFactorModule": {
            "children": [],
            "type": "pytorch"  # Regular PyTorch module
        },
        "ContactConditioning": {
            "children": [],
            "type": "pytorch"  # Regular PyTorch module
        },
        "TemplateModule": {
            "children": [],
            "type": "pytorch"  # Regular PyTorch module
        },
        "TemplateV2Module": {
            "children": [],
            "type": "pytorch"  # Regular PyTorch module
        },
        "DiffusionConditioning": {
            "children": ["PairwiseConditioning", "AtomEncoder"],
            "type": "pytorch"  # Regular PyTorch module
        },
        "AffinityModule": {
            "children": [],
            "type": "pytorch"  # Regular PyTorch module
        }
    }
    
    # Initialize hierarchy with base structure
    for module_name, config in base_hierarchy.items():
        module_info = init_module(module_name, config["type"])
        for child in config["children"]:
            child_info = init_module(child)
            module_info["children"].add(child)
            child_info["parent"] = module_name
    
    # First pass: Collect ONLY modules that have direct timing data
    for func_name, stats in timing_stats.items():
        if not stats["calls"]:
            continue
            
        parts = func_name.split('.')
        module_name = parts[0]
        method_name = parts[-1].replace('()', '')
        
        # Determine module type from timing stats
        is_ttnn = stats.get("is_ttnn", False)
        
        # Additional TTNN detection for modules defined in tenstorrent.py
        if not is_ttnn and module_name in [
            "PairWeightedAveraging", "OuterProductMean", "TriangleAttention",
            "TriangleMultiplication", "TriangleMultiplicationOutgoing", 
            "TriangleMultiplicationIncoming", "TriangleAttentionStartingNode",
            "TriangleAttentionEndingNode", "AttentionPairBias", "AdaLN",
            "ConditionedTransitionBlock", "DiffusionTransformerLayer",
            "MSALayer", "PairformerLayer", "Transition"
        ]:
            is_ttnn = True
            
        module_type = "ttnn" if is_ttnn else "pytorch"
        
        # Initialize module with direct timing
        module_info = init_module(module_name, module_type)
        module_info["total_time"] += stats["total_time"]
        module_info["calls"] += stats["calls"]
        module_info["methods"].add(method_name)
        module_info["direct_timing"] = True
    
    # Second pass: Build parent-child relationships ONLY from known wrapper patterns
    wrapper_patterns = {
        "MSAModule": "MSA",
        "PairformerModule": "Pairformer", 
        "DiffusionTransformerModule": "DiffusionTransformer",
        "DiffusionModule": "DiffusionTransformerModule",
        "AtomDiffusion": "DiffusionModule",
    }
    
    for parent, child in wrapper_patterns.items():
        # Only establish relationship if BOTH modules actually exist in timing data
        if parent in hierarchy and child in hierarchy:
            hierarchy[parent]["children"].add(child)
            hierarchy[child]["parent"] = parent
            
            # Determine hybrid vs pure types
            if hierarchy[child]["type"] == "ttnn" and "forward" in hierarchy[parent]["methods"]:
                hierarchy[parent]["type"] = "hybrid"
            elif hierarchy[parent]["type"] == "unknown":
                hierarchy[parent]["type"] = "pytorch"
    
    # Third pass: Add sub-component relationships from nested timing calls  
    for func_name in timing_stats:
        parts = func_name.split('.')
        if len(parts) >= 2:  # Has at least Module.SubModule
            parent = parts[0]
            child = parts[1].split('(')[0]  # Remove method name
            
            # Only add if both exist and child doesn't already have a parent
            if (parent in hierarchy and child in hierarchy and 
                hierarchy[child]["parent"] is None and parent != child):
                hierarchy[parent]["children"].add(child)
                hierarchy[child]["parent"] = parent
    
    # Fourth pass: Add known composition relationships that we might have missed
    known_compositions = {
        "ConfidenceModule": ["PairformerModule", "ConfidenceHeads"],
        "DiffusionModule": ["DiffusionTransformerModule"],
        "AtomDiffusion": ["DiffusionModule"], 
        "MSAModule": ["MSA"],
        "PairformerModule": ["Pairformer"],
        "DiffusionTransformerModule": ["DiffusionTransformer"],
        "AtomAttentionEncoder": ["AtomTransformer"],
        "AtomAttentionDecoder": ["AtomTransformer"],
        # Main model relationships (MSAModule is called from Boltz2 main model)
        "Boltz2": ["MSAModule", "PairformerModule", "ConfidenceModule", "AtomDiffusion"],
        "Boltz1": ["MSAModule", "PairformerModule", "ConfidenceModule"],
    }
    
    for parent, children in known_compositions.items():
        if parent in hierarchy:
            for child in children:
                if child in hierarchy and hierarchy[child]["parent"] is None:
                    hierarchy[parent]["children"].add(child)
                    hierarchy[child]["parent"] = parent
    
    # Fifth pass: Final type determination
    for module_name, info in hierarchy.items():
        if info["type"] == "unknown" or info["type"] == "pytorch":
            # Check if has TTNN children
            has_ttnn_children = any(
                hierarchy[child]["type"] == "ttnn" 
                for child in info["children"]
            )
            has_forward = "forward" in info["methods"]
            
            if has_ttnn_children and has_forward:
                info["type"] = "hybrid"
            elif has_forward or info["direct_timing"]:
                info["type"] = "pytorch"


    
    # Calculate totals from timing_stats entries 
    total_ttnn_time = 0
    total_torch_time = 0
    
    # Debug: let's see what we have in timing_stats
    logger.info(f"DEBUG: Found {len(timing_stats)} timing entries")
    top_level_count = sum(1 for stats in timing_stats.values() if stats.get("is_top_level", False))
    logger.info(f"DEBUG: {top_level_count} marked as top_level")
    
    for func_name, stats in timing_stats.items():
        # For now, let's use all entries that have significant time instead of just top_level
        if stats["total_time"] > 0.001:  # More than 1ms
            if stats.get("is_ttnn", False) or stats.get("is_hybrid", False):
                total_ttnn_time += stats["total_time"]
                if stats["total_time"] > 0.1:  # Log significant ones
                    logger.info(f"DEBUG TTNN: {func_name} = {stats['total_time']:.3f}s (top_level={stats.get('is_top_level', False)})")
            else:
                total_torch_time += stats["total_time"]
                if stats["total_time"] > 0.1:  # Log significant ones
                    logger.info(f"DEBUG PyTorch: {func_name} = {stats['total_time']:.3f}s (top_level={stats.get('is_top_level', False)})")
    
    # Add conversion times
    total_to_ttnn_time = conversion_stats["torch_to_ttnn"]["total_time"]
    total_to_torch_time = conversion_stats["ttnn_to_torch"]["total_time"]
    total_conversion_time = total_to_ttnn_time + total_to_torch_time
    
    # Total time is sum of all components
    total_time = total_ttnn_time + total_torch_time + total_conversion_time

    # Print timing summary
    logger.info("\n=== HIERARCHICAL MODULE TIMING ===")
    # Find max width needed for the longest module name in hierarchy
    max_name_width = max(60, max(len(f"{name}") + 4 * depth for name, info in hierarchy.items() for depth in [0]))  # 4 chars per depth level for tree chars
    logger.info(f"{'Module':<{max_name_width}} {'Type':<10} {'Time(s)':<10} {'Conv(s)':<10} {'Calls':<8} {'Avg(ms)':<8}")
    logger.info(f"{' ' * (max_name_width - 33)}(Conv(s) = TTNN→Torch + Torch→TTNN)")
    logger.info("─" * (max_name_width + 45))

    # Print all top-level modules first
    for name in sorted(hierarchy.keys()):
        if not hierarchy[name].get("parent"):
            print_module_tree(name, hierarchy[name], hierarchy, base_hierarchy=base_hierarchy)

    # Print summary
    logger.info("\n=== SUMMARY ===")
    logger.info(f"Total Time:      {total_time:>7.3f}s")
    logger.info(f"├── PyTorch:     {total_torch_time:>7.3f}s ({total_torch_time/total_time*100:>4.1f}%)")
    logger.info(f"├── TTNN:        {total_ttnn_time:>7.3f}s ({total_ttnn_time/total_time*100:>4.1f}%)")
    logger.info(f"└── Conversion:  {total_conversion_time:>7.3f}s ({total_conversion_time/total_time*100:>4.1f}%)")
    logger.info(f"    ├── To TTNN:   {total_to_ttnn_time:>7.3f}s")
    logger.info(f"    └── To PyTorch: {total_to_torch_time:>7.3f}s")
    
    # Debug: Show per-module conversion breakdown
    logger.info("\n=== CONVERSION BREAKDOWN ===")
    total_module_conv = 0
    for direction in ["torch_to_ttnn", "ttnn_to_torch"]:
        if conversion_stats[direction]["modules"]:
            logger.info(f"{direction.upper()}:")
            for module_name, stats in conversion_stats[direction]["modules"].items():
                if stats["total_time"] > 0:
                    logger.info(f"  {module_name:<30} {stats['total_time']:>7.3f}s ({stats['calls']:>4} calls)")
                    total_module_conv += stats["total_time"]
    
    unattributed_conv = total_conversion_time - total_module_conv
    logger.info(f"Total attributed to modules: {total_module_conv:>7.3f}s")
    logger.info(f"Unattributed conversion:     {unattributed_conv:>7.3f}s")



def print_module_tree(name, config, hierarchy, depth=0, parent_stats=None, is_last_child=True, parent_prefixes="", base_hierarchy=None):
    # Create visual tree structure with vertical lines
    if depth == 0:
        prefix = ""
        current_prefix = ""
    else:
        if is_last_child:
            prefix = parent_prefixes + "└─ "
            current_prefix = parent_prefixes + "   "
        else:
            prefix = parent_prefixes + "├─ "
            current_prefix = parent_prefixes + "│  "
    
    # Get module type from base hierarchy if available
    if base_hierarchy and name in base_hierarchy:
        module_type = f"[{base_hierarchy[name]['type'].title()}]"
    else:
        module_type = "[TTNN]" if config["type"] == "ttnn" else "[PyTorch]"
        if config["type"] == "hybrid":
            module_type = "[Hybrid]"
    
    # Get timing stats
    total = config["total_time"]
    calls = config["calls"]
    avg_ms = (total / calls * 1000) if calls > 0 else 0
    
    # Get conversion stats for this module
    conv_time = 0
    if config["type"] == "hybrid" or (base_hierarchy and name in base_hierarchy and base_hierarchy[name]["type"] == "ttnn"):
        # For hybrid and TTNN modules, aggregate conversion time from all related components
        module_name = name  # The original module name (e.g., PairformerModule)
        impl_name = name.replace("Module", "")  # The TTNN implementation name (e.g., Pairformer)
        
        # Track both conversion directions
        ttnn_to_torch_time = 0
        torch_to_ttnn_time = 0
        
        # 1. Add conversion stats from the module itself
        if module_name in conversion_stats["ttnn_to_torch"]["modules"]:
            ttnn_to_torch_time += conversion_stats["ttnn_to_torch"]["modules"][module_name]["total_time"]
        if module_name in conversion_stats["torch_to_ttnn"]["modules"]:
            torch_to_ttnn_time += conversion_stats["torch_to_ttnn"]["modules"][module_name]["total_time"]
        
        # 2. Add conversion stats from the TTNN implementation
        if impl_name in conversion_stats["ttnn_to_torch"]["modules"]:
            ttnn_to_torch_time += conversion_stats["ttnn_to_torch"]["modules"][impl_name]["total_time"]
        if impl_name in conversion_stats["torch_to_ttnn"]["modules"]:
            torch_to_ttnn_time += conversion_stats["torch_to_ttnn"]["modules"][impl_name]["total_time"]
        
        # 3. Add conversion stats from ALL sub-components
        def collect_child_conversions(module_hierarchy, current_name):
            child_conv_time = 0
            children = module_hierarchy.get("children", set())
            for child_name in children:
                if child_name in hierarchy:
                    # Add direct conversions from this child
                    if child_name in conversion_stats["ttnn_to_torch"]["modules"]:
                        child_conv_time += conversion_stats["ttnn_to_torch"]["modules"][child_name]["total_time"]
                    if child_name in conversion_stats["torch_to_ttnn"]["modules"]:
                        child_conv_time += conversion_stats["torch_to_ttnn"]["modules"][child_name]["total_time"]
                    # Recursively collect from grandchildren
                    child_conv_time += collect_child_conversions(hierarchy[child_name], child_name)
            return child_conv_time
        
        child_conversions = collect_child_conversions(config, name)
        conv_time = ttnn_to_torch_time + torch_to_ttnn_time + child_conversions
    
    # Print this module if it has timing data or is in base hierarchy
    if total > 0 or name in base_hierarchy or (parent_stats and parent_stats["total_time"] > 0):
        name_with_prefix = f"{prefix}{name}"
        # Find max width needed for the longest module name
        max_name_width = max(60, len(name_with_prefix))  # At least 60 chars for standard cases
        logger.info(f"{name_with_prefix:<{max_name_width}} {module_type:<10} {total:>9.3f} {conv_time:>9.3f} {calls:>7} {avg_ms:>7.1f}")
    
    # Process children with proper tree structure
    children = sorted(config.get("children", set()))
    for i, child_name in enumerate(children):
        if child_name in hierarchy:
            is_last = (i == len(children) - 1)
            print_module_tree(child_name, hierarchy[child_name], hierarchy, depth + 1, config, is_last, current_prefix, base_hierarchy)
















def cleanup():
    global device
    if device is not None:
        print_timing_summary()  # Print timing summary before cleanup
        ttnn.DumpDeviceProfiler(device)
        ttnn.close_device(device)

atexit.register(cleanup)


def filter_dict(state_dict: dict, prefix: str, remove: str = "") -> dict:
    if not prefix:
        return state_dict
    prefix += "."
    return {
        key[len(prefix) :].replace(remove, ""): value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }


class Module:
    def __init__(
        self,
        state_dict: dict,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        self.state_dict = state_dict
        self.compute_kernel_config = compute_kernel_config
    def torch_to_tt(
        self,
        key: str,
        transform: Callable[[torch.Tensor], torch.Tensor] = lambda x: x.t(),
    ) -> ttnn.Tensor:
        start_time = time.time()
        result = ttnn.from_torch(
            transform(self.state_dict[key]),
            layout=ttnn.TILE_LAYOUT,
            device=device,
            dtype=ttnn.float32 if USE_FLOAT32 else ttnn.bfloat16,
        )
        elapsed_time = time.time() - start_time
        
        # Update global stats
        conversion_stats["torch_to_ttnn"]["total_time"] += elapsed_time
        conversion_stats["torch_to_ttnn"]["calls"] += 1
        
        # Update per-module stats
        module_name = self.__class__.__name__.replace("Module", "")
        conversion_stats["torch_to_ttnn"]["modules"][module_name]["total_time"] += elapsed_time
        conversion_stats["torch_to_ttnn"]["modules"][module_name]["calls"] += 1
        
        return result























def cleanup():
    global device
    if device is not None:
        print_timing_summary()  # Print timing summary before cleanup
        ttnn.DumpDeviceProfiler(device)
        ttnn.close_device(device)


atexit.register(cleanup)


def filter_dict(state_dict: dict, prefix: str, remove: str = "") -> dict:
    if not prefix:
        return state_dict
    prefix += "."
    return {
        key[len(prefix) :].replace(remove, ""): value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }


class Module:
    def __init__(
        self,
        state_dict: dict,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        self.state_dict = state_dict
        self.compute_kernel_config = compute_kernel_config

    def torch_to_tt(
        self,
        key: str,
        transform: Callable[[torch.Tensor], torch.Tensor] = lambda x: x.t(),
    ) -> ttnn.Tensor:
        start_time = time.time()
        result = ttnn.from_torch(
            transform(self.state_dict[key]),
            layout=ttnn.TILE_LAYOUT,
            device=device,
            dtype=ttnn.float32 if USE_FLOAT32 else ttnn.bfloat16,
        )
        elapsed_time = time.time() - start_time
        
        # Update global stats
        conversion_stats["torch_to_ttnn"]["total_time"] += elapsed_time
        conversion_stats["torch_to_ttnn"]["calls"] += 1
        
        # Update per-module stats
        module_name = self.__class__.__name__.replace("Module", "")
        conversion_stats["torch_to_ttnn"]["modules"][module_name]["total_time"] += elapsed_time
        conversion_stats["torch_to_ttnn"]["modules"][module_name]["calls"] += 1
        
        return result


class TriangleMultiplication(Module):
    def __init__(
        self,
        ending: bool,
        state_dict: dict,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.ending = ending
        self.in_norm_weight = self.torch_to_tt("norm_in.weight")
        self.in_norm_bias = self.torch_to_tt("norm_in.bias")
        self.out_norm_weight = self.torch_to_tt("norm_out.weight")
        self.out_norm_bias = self.torch_to_tt("norm_out.bias")
        self.out_p = self.torch_to_tt("p_out.weight")
        self.gpg_weight = ttnn.from_torch(
            torch.cat(
                [
                    self.state_dict["g_in.weight"],
                    self.state_dict["p_in.weight"],
                    self.state_dict["g_out.weight"],
                    torch.zeros_like(torch.zeros_like(self.state_dict["g_out.weight"])),
                ],
                dim=0,
            ).t(),
            layout=ttnn.TILE_LAYOUT,
            device=device,
            dtype=ttnn.float32 if USE_FLOAT32 else ttnn.bfloat8_b,
        )

    @timing_decorator
    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x_norm_in = ttnn.layer_norm(
            x,
            weight=self.in_norm_weight,
            bias=self.in_norm_bias,
            epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )
        gpg_in = ttnn.linear(
            x_norm_in,
            self.gpg_weight,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            core_grid=ttnn.CoreGrid(y=10, x=13) if is_blackhole() else None,
        )
        g_in, p_in, g_out = ttnn.experimental.nlp_create_qkv_heads_boltz(
            gpg_in,
            num_heads=1,
            num_kv_heads=1,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        g_in = ttnn.sigmoid_accurate(g_in)
        x_pg_in = ttnn.multiply(p_in, g_in, dtype=ttnn.bfloat16)
        B, H, H, W = x_pg_in.shape
        for chunk_start in range(0, W // 2, TRIANGLE_MULT_CHUNK_SIZE):
            a_chunk = ttnn.slice(
                x_pg_in,
                [0, 0, 0, chunk_start],
                [B, H, H, chunk_start + TRIANGLE_MULT_CHUNK_SIZE],
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            a_chunk = ttnn.permute(
                a_chunk, (0, 3) + ((2, 1) if self.ending else (1, 2))
            )
            a_chunk = ttnn.typecast(a_chunk, ttnn.bfloat8_b)
            a_chunk = ttnn.reallocate(a_chunk)
            b_chunk = ttnn.slice(
                x_pg_in,
                [0, 0, 0, W // 2 + chunk_start],
                [B, H, H, W // 2 + chunk_start + TRIANGLE_MULT_CHUNK_SIZE],
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            b_chunk = ttnn.permute(
                b_chunk, (0, 3) + ((1, 2) if self.ending else (2, 1))
            )
            b_chunk = ttnn.typecast(b_chunk, ttnn.bfloat8_b)
            b_chunk = ttnn.reallocate(b_chunk)
            x_chunk = ttnn.matmul(
                a_chunk,
                b_chunk,
                compute_kernel_config=self.compute_kernel_config,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                core_grid=ttnn.CoreGrid(y=9, x=2) if is_blackhole() else None,
            )
            ttnn.deallocate(a_chunk)
            ttnn.deallocate(b_chunk)
            x_chunk = ttnn.permute(x_chunk, (0, 2, 3, 1))
            if chunk_start == 0:
                x = ttnn.clone(x_chunk, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            else:
                x = ttnn.concat([x, x_chunk], dim=-1)
            ttnn.deallocate(x_chunk)
        x_norm_out = ttnn.layer_norm(
            x,
            weight=self.out_norm_weight,
            bias=self.out_norm_bias,
            epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )
        p_out = ttnn.linear(
            x_norm_out,
            self.out_p,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            core_grid=ttnn.CoreGrid(y=10, x=11) if is_blackhole() else None,
        )
        g_out = ttnn.sigmoid_accurate(g_out[:, :, :, : W // 2])
        x = ttnn.multiply_(p_out, g_out)
        return x


class TriangleAttention(Module):
    def __init__(
        self,
        head_dim: int,
        n_heads: int,
        ending: bool,
        state_dict: dict,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.ending = ending
        self.layer_norm_weight = self.torch_to_tt("layer_norm.weight")
        self.layer_norm_bias = self.torch_to_tt("layer_norm.bias")
        self.bias_weight = self.torch_to_tt("linear.weight")
        self.o_weight = self.torch_to_tt("linear_o.weight")
        self.qkvg_weight = ttnn.from_torch(
            torch.cat(
                [
                    self.state_dict["linear_q.weight"],
                    self.state_dict["linear_k.weight"],
                    self.state_dict["linear_v.weight"],
                    self.state_dict["linear_g.weight"],
                ],
                dim=0,
            ).t(),
            layout=ttnn.TILE_LAYOUT,
            device=device,
            dtype=ttnn.float32 if USE_FLOAT32 else ttnn.bfloat8_b,
        )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = ttnn.reshape(x, tuple(x.shape)[1:])
        if self.ending:
            x = ttnn.permute(x, (1, 0, 2))  # THIS CAUSES CACHE -> RESHAPE PROBLEM
        x = ttnn.layer_norm(
            x,
            weight=self.layer_norm_weight,
            bias=self.layer_norm_bias,
            epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        seq_len = x.shape[0]
        padding = -seq_len % 256
        x = ttnn.pad(x, [(0, padding), (0, padding), (0, 0)], 0)
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        triangle_bias = ttnn.linear(
            x,
            self.bias_weight,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat8_b,
            core_grid=ttnn.CoreGrid(y=9, x=12) if is_blackhole() else None,
        )
        triangle_bias = ttnn.reshape(triangle_bias, (1, *triangle_bias.shape))
        triangle_bias = ttnn.permute(triangle_bias, (3, 0, 1, 2))
        qkvg = ttnn.linear(
            x,
            self.qkvg_weight,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat8_b,
            core_grid=ttnn.CoreGrid(y=10, x=12) if is_blackhole() else None,
        )
        split_idx = 3 * self.head_dim * self.n_heads
        qkv = qkvg[:, :, :split_idx]
        g = qkvg[:, :, split_idx:]
        del qkvg
        qkv = ttnn.unsqueeze(qkv, 0)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_boltz(
            qkv,
            num_heads=self.n_heads,
            num_kv_heads=self.n_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for head in range(0, q.shape[0]):
            head_q = q[head : head + 1, :, :, :]
            head_k = k[head : head + 1, :, :, :]
            head_v = v[head : head + 1, :, :, :]
            head_triangle_bias = triangle_bias[head : head + 1, :, :, :]
            head_o = ttnn.transformer.scaled_dot_product_attention(
                head_q,
                head_k,
                head_v,
                attn_mask=head_triangle_bias,
                is_causal=False,
                scale=self.head_dim**-0.5,
                program_config=ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=(
                        (13, 10) if is_blackhole() else (8, 8)
                    ),
                    exp_approx_mode=False,
                    q_chunk_size=256,
                    k_chunk_size=256,
                ),
            )
            if head == 0:
                o = head_o
            else:
                o = ttnn.concat([o, head_o], dim=0)
        o = ttnn.experimental.nlp_concat_heads_boltz(
            o, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        o = ttnn.squeeze(o, 0)
        g = ttnn.sigmoid_accurate(g)
        o = ttnn.multiply(o, g)
        x = ttnn.linear(
            o,
            self.o_weight,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat8_b,
            core_grid=ttnn.CoreGrid(y=6, x=12) if is_blackhole() else None,
        )
        x = x[:seq_len, :seq_len, :]
        if self.ending:
            x = ttnn.permute(x, (1, 0, 2))
        x = ttnn.reshape(x, (1, *x.shape))
        return x


class AttentionPairBias(Module):
    def __init__(
        self,
        head_dim: int,
        n_heads: int,
        compute_pair_bias: bool,
        atom_level: bool,
        state_dict: dict,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.compute_pair_bias = compute_pair_bias
        self.atom_level = atom_level
        self.q_weight = self.torch_to_tt("proj_q.weight")
        self.q_bias = self.torch_to_tt("proj_q.bias")
        self.k_weight = self.torch_to_tt("proj_k.weight")
        self.v_weight = self.torch_to_tt("proj_v.weight")
        self.g_weight = self.torch_to_tt("proj_g.weight")
        if compute_pair_bias:
            self.z_norm_weight = self.torch_to_tt("proj_z.0.weight")
            self.z_norm_bias = self.torch_to_tt("proj_z.0.bias")
            self.z_weight = self.torch_to_tt("proj_z.1.weight")
        self.o_weight = self.torch_to_tt("proj_o.weight")

    @timing_decorator
    def __call__(
        self,
        s: ttnn.Tensor,
        z: ttnn.Tensor,
        s_kv: ttnn.Tensor = None,
    ) -> ttnn.Tensor:
        memory_config = ttnn.L1_MEMORY_CONFIG if self.atom_level else None
        if not self.atom_level:
            s_kv = s
        q = ttnn.linear(
            s,
            self.q_weight,
            bias=self.q_bias,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=memory_config,
        )
        k = ttnn.linear(
            s_kv,
            self.k_weight,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=memory_config,
        )
        v = ttnn.linear(
            s_kv,
            self.v_weight,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=memory_config,
        )
        q = ttnn.permute(q, (2, 0, 1))
        k = ttnn.permute(k, (2, 0, 1))
        v = ttnn.permute(v, (2, 0, 1))
        q = ttnn.reshape(q, (self.n_heads, self.head_dim, *tuple(q.shape)[1:]))
        k = ttnn.reshape(k, (self.n_heads, self.head_dim, *tuple(k.shape)[1:]))
        v = ttnn.reshape(v, (self.n_heads, self.head_dim, *tuple(v.shape)[1:]))
        q = ttnn.permute(q, (0, 2, 3, 1))
        k = ttnn.permute(k, (0, 2) + ((1, 3) if self.atom_level else (3, 1)))
        v = ttnn.permute(v, (0, 2, 3, 1))
        if not self.atom_level:
            seq_len = s.shape[1]
            seq_len_padding = -seq_len % 32
            if self.compute_pair_bias:
                z = ttnn.layer_norm(
                    z,
                    weight=self.z_norm_weight,
                    bias=self.z_norm_bias,
                    epsilon=1e-5,
                    compute_kernel_config=self.compute_kernel_config,
                )
                z = ttnn.linear(
                    z,
                    self.z_weight,
                    compute_kernel_config=self.compute_kernel_config,
                    core_grid=ttnn.CoreGrid(y=8, x=11) if is_blackhole() else None,
                )
                z = ttnn.permute(z, (3, 0, 1, 2))
                z = ttnn.pad(
                    z, [(0, 0), (0, 0), (0, seq_len_padding), (0, seq_len_padding)], 0
                )
            head_dim = q.shape[-1]
            head_dim_padding = -head_dim % 32
            q = ttnn.pad(
                q, [(0, 0), (0, 0), (0, seq_len_padding), (0, head_dim_padding)], 0
            )
            k = ttnn.pad(
                k, [(0, 0), (0, 0), (0, seq_len_padding), (0, head_dim_padding)], 0
            )
            v = ttnn.pad(
                v, [(0, 0), (0, 0), (0, seq_len_padding), (0, head_dim_padding)], 0
            )
            o = ttnn.transformer.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=z,
                is_causal=False,
                scale=self.head_dim**-0.5,
                program_config=ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=(
                        (13, 10) if is_blackhole() else (8, 8)
                    ),
                    exp_approx_mode=False,
                    q_chunk_size=32,
                    k_chunk_size=32,
                ),
            )
            o = o[:, :, :seq_len, :head_dim]
        else:
            a = ttnn.matmul(
                q,
                k,
                compute_kernel_config=self.compute_kernel_config,
                memory_config=memory_config,
            )
            ttnn.deallocate(q)
            ttnn.deallocate(k)
            a = ttnn.multiply_(a, self.head_dim**-0.5)
            a = ttnn.add_(a, z)
            a = ttnn.softmax(
                a,
                dim=-1,
                compute_kernel_config=self.compute_kernel_config,
                numeric_stable=True,
            )
            o = ttnn.matmul(
                a,
                v,
                compute_kernel_config=self.compute_kernel_config,
                memory_config=memory_config,
            )
            ttnn.deallocate(a)
            ttnn.deallocate(v)
        o = ttnn.permute(o, (0, 3, 1, 2))
        o = ttnn.reshape(o, (-1, *tuple(o.shape)[2:]))
        o = ttnn.permute(o, (1, 2, 0))
        g = ttnn.linear(
            s,
            self.g_weight,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=memory_config,
        )
        g = ttnn.sigmoid_accurate(g)
        o = ttnn.multiply_(o, g)
        if self.atom_level:
            ttnn.deallocate(g)
        x = ttnn.linear(
            o, self.o_weight, compute_kernel_config=self.compute_kernel_config
        )
        return x


class Transition(Module):
    def __init__(
        self,
        chunking: bool,
        state_dict: dict,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.norm_weight = self.torch_to_tt("norm.weight")
        self.norm_bias = self.torch_to_tt("norm.bias")
        self.fc1_weight = self.torch_to_tt("fc1.weight")
        self.fc2_weight = self.torch_to_tt("fc2.weight")
        self.fc3_weight = self.torch_to_tt("fc3.weight")
        self.chunking = chunking

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        def f(x):
            x_norm = ttnn.layer_norm(
                x,
                weight=self.norm_weight,
                bias=self.norm_bias,
                epsilon=1e-5,
                compute_kernel_config=self.compute_kernel_config,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            x_1 = ttnn.linear(
                x_norm,
                self.fc1_weight,
                activation="silu",
                compute_kernel_config=self.compute_kernel_config,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                dtype=ttnn.bfloat8_b,
                core_grid=ttnn.CoreGrid(y=10, x=13) if is_blackhole() else None,
            )
            x_2 = ttnn.linear(
                x_norm,
                self.fc2_weight,
                compute_kernel_config=self.compute_kernel_config,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                dtype=ttnn.bfloat8_b,
                core_grid=ttnn.CoreGrid(y=10, x=13) if is_blackhole() else None,
            )
            ttnn.deallocate(x_norm)
            x = ttnn.multiply(x_1, x_2, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(x_1)
            ttnn.deallocate(x_2)
            x = ttnn.linear(
                x,
                self.fc3_weight,
                compute_kernel_config=self.compute_kernel_config,
                dtype=ttnn.bfloat16,
                core_grid=ttnn.CoreGrid(y=8, x=11) if is_blackhole() else None,
            )
            return x

        if not self.chunking:
            x = f(x)
        else:
            for chunk_start in range(0, x.shape[1], TRANSITION_CHUNK_SIZE):
                x_chunk = x[
                    :,
                    chunk_start : min(chunk_start + TRANSITION_CHUNK_SIZE, x.shape[1]),
                    :,
                    :,
                ]
                x_chunk = f(x_chunk)
                if chunk_start == 0:
                    x_out = x_chunk
                else:
                    x_out = ttnn.concat([x_out, x_chunk], dim=1)
            x = x_out
        return x


class PairformerLayer(Module):
    def __init__(
        self,
        tri_att_head_dim: int,
        tri_att_n_heads: int,
        att_head_dim: int,
        att_n_heads: int,
        transform_s: bool,
        state_dict: dict,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.transform_s = transform_s
        self.triangle_multiplication_start = TriangleMultiplication(
            False, filter_dict(state_dict, "tri_mul_out"), compute_kernel_config
        )
        self.triangle_multiplication_end = TriangleMultiplication(
            True, filter_dict(state_dict, "tri_mul_in"), compute_kernel_config
        )
        self.triangle_attention_start = TriangleAttention(
            tri_att_head_dim,
            tri_att_n_heads,
            False,
            filter_dict(state_dict, "tri_att_start", "mha."),
            compute_kernel_config,
        )
        self.triangle_attention_end = TriangleAttention(
            tri_att_head_dim,
            tri_att_n_heads,
            True,
            filter_dict(state_dict, "tri_att_end", "mha."),
            compute_kernel_config,
        )
        self.transition_z = Transition(
            True, filter_dict(state_dict, "transition_z"), compute_kernel_config
        )
        if transform_s:
            self.pre_norm_s_weight = self.torch_to_tt("pre_norm_s.weight")
            self.pre_norm_s_bias = self.torch_to_tt("pre_norm_s.bias")
            self.attention_pair_bias = AttentionPairBias(
                att_head_dim,
                att_n_heads,
                True,
                False,
                filter_dict(state_dict, "attention"),
                compute_kernel_config,
            )
            self.transition_s = Transition(
                False, filter_dict(state_dict, "transition_s"), compute_kernel_config
            )

    @timing_decorator
    def __call__(
        self, s: ttnn.Tensor, z: ttnn.Tensor
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        z = ttnn.add(
            z,
            self.triangle_multiplication_start(z),
        )
        z = ttnn.add(
            z,
            self.triangle_multiplication_end(z),
        )
        z = ttnn.add(
            z,
            self.triangle_attention_start(z),
        )
        z = ttnn.add(
            z,
            self.triangle_attention_end(z),
        )
        z = ttnn.add(z, self.transition_z(z))
        if self.transform_s:
            s_norm = ttnn.layer_norm(
                s,
                weight=self.pre_norm_s_weight,
                bias=self.pre_norm_s_bias,
                epsilon=1e-5,
                compute_kernel_config=self.compute_kernel_config,
            )
            s = ttnn.add(
                s,
                self.attention_pair_bias(
                    s_norm,
                    z,
                ),
            )
            s = ttnn.add(s, self.transition_s(s))
        return s, z


class Pairformer(Module):
    def __init__(
        self,
        n_blocks: int,
        tri_att_head_dim: int,
        tri_att_n_heads: int,
        att_head_dim: int,
        att_n_heads: int,
        transform_s: bool,
        state_dict: dict,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.blocks = [
            PairformerLayer(
                tri_att_head_dim,
                tri_att_n_heads,
                att_head_dim,
                att_n_heads,
                transform_s,
                filter_dict(state_dict, f"layers.{i}"),
                compute_kernel_config,
            )
            for i in range(n_blocks)
        ]

    @timing_decorator
    def __call__(
        self, s: ttnn.Tensor, z: ttnn.Tensor
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        for block in self.blocks:
            s, z = block(s, z)
        return s, z


class AdaLN(Module):
    def __init__(
        self,
        atom_level: bool,
        state_dict: dict,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.atom_level = atom_level
        self.s_norm_weight = self.torch_to_tt("s_norm.weight")
        self.s_scale_weight = self.torch_to_tt("s_scale.weight")
        self.s_scale_bias = self.torch_to_tt("s_scale.bias")
        self.s_bias_weight = self.torch_to_tt("s_bias.weight")

    def __call__(self, a: ttnn.Tensor, s: ttnn.Tensor) -> ttnn.Tensor:
        memory_config = ttnn.L1_MEMORY_CONFIG if self.atom_level else None
        if not USE_FLOAT32:
            a = ttnn.clone(a, dtype=ttnn.float32, memory_config=memory_config)
            s = ttnn.clone(s, dtype=ttnn.float32, memory_config=memory_config)
        a = ttnn.layer_norm(
            a, epsilon=1e-5, compute_kernel_config=self.compute_kernel_config
        )
        s = ttnn.layer_norm(
            s,
            weight=self.s_norm_weight,
            epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )
        if not USE_FLOAT32:
            a = ttnn.clone(a, dtype=ttnn.bfloat16, memory_config=memory_config)
            s = ttnn.clone(s, dtype=ttnn.bfloat16, memory_config=memory_config)
        s_scale = ttnn.linear(
            s,
            self.s_scale_weight,
            bias=self.s_scale_bias,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=memory_config,
        )
        s_scale = ttnn.sigmoid_accurate(s_scale)
        s_bias = ttnn.linear(
            s,
            self.s_bias_weight,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=memory_config,
        )
        a = ttnn.multiply_(a, s_scale)
        a = ttnn.add_(a, s_bias)
        return a


class ConditionedTransitionBlock(Module):
    def __init__(
        self,
        atom_level: bool,
        state_dict: dict,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.atom_level = atom_level
        self.adaln = AdaLN(
            atom_level, filter_dict(state_dict, "adaln"), compute_kernel_config
        )
        self.swish_weight = self.torch_to_tt("swish_gate.0.weight")
        self.a_to_b_weight = self.torch_to_tt("a_to_b.weight")
        self.b_to_a_weight = self.torch_to_tt("b_to_a.weight")
        self.output_projection_weight = self.torch_to_tt("output_projection.0.weight")
        self.output_projection_bias = self.torch_to_tt("output_projection.0.bias")

    def __call__(self, a: ttnn.Tensor, s: ttnn.Tensor) -> ttnn.Tensor:
        memory_config = ttnn.L1_MEMORY_CONFIG if self.atom_level else None
        a = self.adaln(a, s)
        a_swish = ttnn.linear(
            a,
            self.swish_weight,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=memory_config,
            core_grid=ttnn.CoreGrid(y=10, x=13) if is_blackhole() else None,
        )
        dim = int(a_swish.shape[-1] / 2)
        a_swish, gates = a_swish[:, :, :dim], a_swish[:, :, dim:]
        gates = ttnn.silu(gates)
        a_swish = ttnn.multiply_(gates, a_swish)
        a_b = ttnn.linear(
            a,
            self.a_to_b_weight,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=memory_config,
        )
        b = ttnn.multiply_(a_swish, a_b)
        if self.atom_level:
            ttnn.deallocate(a_b)
        s = ttnn.linear(
            s,
            self.output_projection_weight,
            bias=self.output_projection_bias,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=memory_config,
        )
        s = ttnn.sigmoid_accurate(s)
        b_a = ttnn.linear(
            b,
            self.b_to_a_weight,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=memory_config,
        )
        if self.atom_level:
            ttnn.deallocate(b)
        a = ttnn.multiply_(s, b_a)
        return a


class DiffusionTransformerLayer(Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        atom_level: bool,
        state_dict: dict,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.atom_level = atom_level
        self.adaln = AdaLN(
            atom_level, filter_dict(state_dict, "adaln"), compute_kernel_config
        )
        self.attn_pair_bias = AttentionPairBias(
            head_dim=dim // n_heads,
            n_heads=n_heads,
            compute_pair_bias=False,
            atom_level=atom_level,
            state_dict=filter_dict(state_dict, "pair_bias_attn"),
            compute_kernel_config=compute_kernel_config,
        )
        self.output_projection_weight = self.torch_to_tt(
            "output_projection_linear.weight"
        )
        self.output_projection_bias = self.torch_to_tt("output_projection_linear.bias")
        self.transition = ConditionedTransitionBlock(
            atom_level,
            filter_dict(state_dict, "transition"),
            compute_kernel_config,
        )

    @timing_decorator
    def __call__(
        self,
        a: ttnn.Tensor,
        s: ttnn.Tensor,
        z: ttnn.Tensor,
        keys_indexing: ttnn.Tensor,
    ) -> ttnn.Tensor:
        b = self.adaln(a, s)
        if not self.atom_level:
            b = self.attn_pair_bias(b, z)
        else:
            K, W, D = b.shape
            b_kv = ttnn.reshape(b, (2 * K, W // 2, -1))
            b_kv = ttnn.permute(b_kv, (1, 2, 0))
            b_kv = ttnn.matmul(
                b_kv,
                keys_indexing,
                compute_kernel_config=self.compute_kernel_config,
                core_grid=ttnn.CoreGrid(y=10, x=13) if is_blackhole() else None,
            )
            b_kv = ttnn.permute(b_kv, (2, 0, 1))
            b_kv = ttnn.reshape(b_kv, (K, -1, D))
            b = self.attn_pair_bias(b, z, b_kv)
        s_o = ttnn.linear(
            s,
            self.output_projection_weight,
            bias=self.output_projection_bias,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=ttnn.CoreGrid(y=10, x=13) if is_blackhole() else None,
        )
        s_o = ttnn.sigmoid_accurate(s_o)
        b = ttnn.multiply(s_o, b)
        a = ttnn.add(a, b)
        a_t = self.transition(a, s)
        a = ttnn.add(a, a_t)
        return a


class DiffusionTransformer(Module):
    def __init__(
        self,
        n_layers: int,
        dim: int,
        n_heads: int,
        atom_level: bool,
        state_dict: dict,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.layers = [
            DiffusionTransformerLayer(
                dim,
                n_heads,
                atom_level,
                filter_dict(state_dict, f"layers.{i}"),
                compute_kernel_config,
            )
            for i in range(n_layers)
        ]

    @timing_decorator
    def __call__(
        self,
        a: ttnn.Tensor,
        s: ttnn.Tensor,
        z: ttnn.Tensor,
        keys_indexing: ttnn.Tensor,
    ) -> ttnn.Tensor:
        dim = z.shape[0] // len(self.layers)
        for i, layer in enumerate(self.layers):
            a = layer(a, s, z[i * dim : (i + 1) * dim, :, :, :], keys_indexing)
        return a


class PairWeightedAveraging(Module):
    def __init__(
        self,
        head_dim: int,
        n_heads: int,
        state_dict: dict,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.m_norm_weight = self.torch_to_tt("norm_m.weight")
        self.m_norm_bias = self.torch_to_tt("norm_m.bias")
        self.z_norm_weight = self.torch_to_tt("norm_z.weight")
        self.z_norm_bias = self.torch_to_tt("norm_z.bias")
        self.m_weight = self.torch_to_tt("proj_m.weight")
        self.g_weight = self.torch_to_tt("proj_g.weight")
        self.z_weight = self.torch_to_tt("proj_z.weight")
        self.o_weight = self.torch_to_tt("proj_o.weight")

    @timing_decorator
    def __call__(self, m: ttnn.Tensor, z: ttnn.Tensor) -> ttnn.Tensor:
        m = ttnn.reshape(m, tuple(m.shape)[1:])
        z = ttnn.reshape(z, tuple(z.shape)[1:])
        m = ttnn.layer_norm(
            m,
            weight=self.m_norm_weight,
            bias=self.m_norm_bias,
            epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )
        z = ttnn.layer_norm(
            z,
            weight=self.z_norm_weight,
            bias=self.z_norm_bias,
            epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )
        for i in range(self.n_heads):
            b = ttnn.linear(
                z,
                self.z_weight[:, i : i + 1],
                compute_kernel_config=self.compute_kernel_config,
                core_grid=ttnn.CoreGrid(y=10, x=13) if is_blackhole() else None,
            )
            b = ttnn.permute(b, (2, 0, 1))
            w = ttnn.softmax(
                b,
                dim=-1,
                compute_kernel_config=self.compute_kernel_config,
                numeric_stable=True,
            )
            v = ttnn.linear(
                m,
                self.m_weight[:, i * self.head_dim : (i + 1) * self.head_dim],
                compute_kernel_config=self.compute_kernel_config,
                core_grid=ttnn.CoreGrid(y=10, x=13) if is_blackhole() else None,
            )

            o = ttnn.matmul(
                v,
                w,
                transpose_a=True,
                transpose_b=True,
                compute_kernel_config=self.compute_kernel_config,
                core_grid=ttnn.CoreGrid(y=10, x=13) if is_blackhole() else None,
            )
            del v, w
            o = ttnn.permute(o, (0, 2, 1))
            g = ttnn.linear(
                m,
                self.g_weight[:, i * self.head_dim : (i + 1) * self.head_dim],
                compute_kernel_config=self.compute_kernel_config,
                core_grid=ttnn.CoreGrid(y=10, x=13) if is_blackhole() else None,
            )
            g = ttnn.sigmoid_accurate(g)
            o = ttnn.multiply(o, g)
            del g
            o = ttnn.linear(
                o,
                self.o_weight[i * self.head_dim : (i + 1) * self.head_dim, :],
                compute_kernel_config=self.compute_kernel_config,
                core_grid=ttnn.CoreGrid(y=10, x=13) if is_blackhole() else None,
            )
            if i == 0:
                o_out = o
            else:
                o_out = ttnn.add(o_out, o)
        o_out = ttnn.reshape(o_out, (1, *o_out.shape))
        return o_out


class OuterProductMean(Module):
    def __init__(
        self,
        state_dict: dict,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.norm_weight = self.torch_to_tt("norm.weight")
        self.norm_bias = self.torch_to_tt("norm.bias")
        self.a_weight = self.torch_to_tt("proj_a.weight")
        self.b_weight = self.torch_to_tt("proj_b.weight")
        self.o_weight = self.torch_to_tt("proj_o.weight")
        self.o_bias = self.torch_to_tt("proj_o.bias")

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = ttnn.reshape(x, tuple(x.shape)[1:])
        m = ttnn.layer_norm(
            x,
            weight=self.norm_weight,
            bias=self.norm_bias,
            epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )
        a = ttnn.linear(
            m,
            self.a_weight,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=ttnn.CoreGrid(y=10, x=13) if is_blackhole() else None,
        )
        b = ttnn.linear(
            m,
            self.b_weight,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=ttnn.CoreGrid(y=10, x=13) if is_blackhole() else None,
        )
        S, I, C = a.shape
        _, J, D = b.shape
        a = ttnn.permute(a, (1, 2, 0))
        a = ttnn.reshape(a, (-1, S))
        b = ttnn.permute(b, (0, 2, 1))
        b = ttnn.reshape(b, (S, -1))
        z = ttnn.matmul(a, b, compute_kernel_config=self.compute_kernel_config)
        z = ttnn.to_layout(z, ttnn.ROW_MAJOR_LAYOUT)
        z = ttnn.reshape(z, (I, C * D, J))
        z = ttnn.to_layout(z, ttnn.TILE_LAYOUT)
        z = ttnn.permute(z, (0, 2, 1))
        z = ttnn.multiply(z, 1 / S)
        z = ttnn.linear(
            z,
            self.o_weight,
            bias=self.o_bias,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=ttnn.CoreGrid(y=10, x=13) if is_blackhole() else None,
        )
        z = ttnn.reshape(z, (1, *z.shape))
        return z


class MSALayer(Module):
    def __init__(
        self,
        avg_head_dim: int,
        avg_n_heads: int,
        tri_att_head_dim: int,
        tri_att_n_heads: int,
        state_dict: dict,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.msa_transition = Transition(
            True, filter_dict(state_dict, "msa_transition"), compute_kernel_config
        )
        self.pair_weighted_averaging = PairWeightedAveraging(
            head_dim=avg_head_dim,
            n_heads=avg_n_heads,
            state_dict=filter_dict(state_dict, "pair_weighted_averaging"),
            compute_kernel_config=compute_kernel_config,
        )
        self.outer_product_mean = OuterProductMean(
            state_dict=filter_dict(state_dict, "outer_product_mean"),
            compute_kernel_config=compute_kernel_config,
        )
        self.pairformer_layer = PairformerLayer(
            tri_att_head_dim,
            tri_att_n_heads,
            None,
            None,
            False,
            filter_dict(state_dict, f"pairformer_layer"),
            compute_kernel_config,
        )

    @timing_decorator
    def __call__(
        self, z: ttnn.Tensor, m: ttnn.Tensor
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        m = ttnn.add(m, self.pair_weighted_averaging(m, z))
        m = ttnn.add(m, self.msa_transition(m))
        z = ttnn.add(z, self.outer_product_mean(m))
        z = self.pairformer_layer(None, z)[1]
        return z, m


class MSA(Module):
    def __init__(
        self,
        n_blocks: int,
        avg_head_dim: int,
        avg_n_heads: int,
        tri_att_head_dim: int,
        tri_att_n_heads: int,
        state_dict: dict,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.s_weight = self.torch_to_tt("s_proj.weight")
        self.msa_weight = self.torch_to_tt("msa_proj.weight")
        self.blocks = [
            MSALayer(
                avg_head_dim,
                avg_n_heads,
                tri_att_head_dim,
                tri_att_n_heads,
                filter_dict(state_dict, f"layers.{i}"),
                compute_kernel_config,
            )
            for i in range(n_blocks)
        ]

    @timing_decorator
    def __call__(self, z: ttnn.Tensor, m: ttnn.Tensor, emb: ttnn.Tensor) -> ttnn.Tensor:
        m = ttnn.linear(
            m,
            self.msa_weight,
            compute_kernel_config=self.compute_kernel_config,
        )
        m = ttnn.add(
            m,
            ttnn.linear(
                emb,
                self.s_weight,
                compute_kernel_config=self.compute_kernel_config,
            ),
        )
        for block in self.blocks:
            z, m = block(z, m)
        return z


class TorchWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = None
        global device
        if device is None:
            ttnn.device.EnablePersistentKernelCache()  # be careful, can lead to bugs when profiling etc.
            args = {"device_id": 0}
            if is_wormhole_b0():
                args["dispatch_core_config"] = ttnn.DispatchCoreConfig(
                    ttnn.device.DispatchCoreType.ETH, ttnn.DispatchCoreAxis.ROW
                )
            device = ttnn.open_device(**args)
            device.enable_program_cache()
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def _from_torch(self, x: torch.Tensor) -> ttnn.Tensor:
        start_time = time.time()
        result = ttnn.from_torch(
            x,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.float32 if USE_FLOAT32 else ttnn.bfloat16,
        )
        elapsed_time = time.time() - start_time
        # Track global stats
        conversion_stats["torch_to_ttnn"]["total_time"] += elapsed_time
        conversion_stats["torch_to_ttnn"]["calls"] += 1
        # Track per-module stats
        module_name = self.__class__.__name__.replace("Module", "")  # e.g., PairformerModule -> Pairformer
        conversion_stats["torch_to_ttnn"]["modules"][module_name]["total_time"] += elapsed_time
        conversion_stats["torch_to_ttnn"]["modules"][module_name]["calls"] += 1
        return result

    def _to_torch(self, x: ttnn.Tensor) -> torch.Tensor:
        start_time = time.time()
        result = torch.Tensor(ttnn.to_torch(x)).to(torch.float32)
        elapsed_time = time.time() - start_time
        # Track global stats
        conversion_stats["ttnn_to_torch"]["total_time"] += elapsed_time
        conversion_stats["ttnn_to_torch"]["calls"] += 1
        # Track per-module stats
        module_name = self.__class__.__name__.replace("Module", "")  # e.g., PairformerModule -> Pairformer
        conversion_stats["ttnn_to_torch"]["modules"][module_name]["total_time"] += elapsed_time
        conversion_stats["ttnn_to_torch"]["modules"][module_name]["calls"] += 1
        return result


class PairformerModule(TorchWrapper):
    def __init__(
        self,
        n_blocks: int,
        tri_att_head_dim: int,
        tri_att_n_heads: int,
        att_head_dim: int,
        att_n_heads: int,
        transform_s: bool,
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.tri_att_head_dim = tri_att_head_dim
        self.tri_att_n_heads = tri_att_n_heads
        self.att_head_dim = att_head_dim
        self.att_n_heads = att_n_heads
        self.transform_s = transform_s

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        self.module = Pairformer(
            self.n_blocks,
            self.tri_att_head_dim,
            self.tri_att_n_heads,
            self.att_head_dim,
            self.att_n_heads,
            self.transform_s,
            filter_dict(state_dict, prefix[:-1]),
            self.compute_kernel_config,
        )

    @timing_decorator
    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor = None,
        pair_mask: torch.Tensor = None,
        use_kernels: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return tuple(
            self._to_torch(x) if x is not None else None
            for x in self.module(
                self._from_torch(s) if s is not None else None,
                self._from_torch(z),
            )
        )


class DiffusionTransformerModule(TorchWrapper):
    def __init__(self, n_layers: int, dim: int, n_heads: int, atom_level: bool):
        super().__init__()
        self.n_layers = n_layers
        self.dim = dim
        self.n_heads = n_heads
        self.atom_level = atom_level
        self.bias = None
        self.keys_indexing = None

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        self.module = DiffusionTransformer(
            self.n_layers,
            self.dim,
            self.n_heads,
            self.atom_level,
            filter_dict(state_dict, prefix[:-1]),
            self.compute_kernel_config,
        )

    @timing_decorator
    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        bias: torch.Tensor,
        mask: torch.Tensor = None,
        keys_indexing: torch.Tensor = None,
        to_keys=None,
        multiplicity: int = 1,
        model_cache: torch.Tensor = None,
    ) -> torch.Tensor:
        if self.bias is None:
            self.bias = self._from_torch(bias.permute(3, 0, 1, 2))
            if not self.atom_level:
                seq_len_padding = -self.bias.shape[-1] % 32
                self.bias = ttnn.pad(
                    self.bias,
                    [(0, 0), (0, 0), (0, seq_len_padding), (0, seq_len_padding)],
                    0,
                )
        if self.atom_level and self.keys_indexing is None:
            self.keys_indexing = self._from_torch(keys_indexing)
            mask = self._from_torch(mask)
            K, W = mask.shape
            mask = ttnn.reshape(mask, (2 * K, W // 2, -1))
            mask = ttnn.permute(mask, (1, 2, 0))
            mask = ttnn.matmul(
                mask,
                self.keys_indexing,
                compute_kernel_config=self.compute_kernel_config,
                core_grid=ttnn.CoreGrid(y=10, x=13) if is_blackhole() else None,
            )
            mask = ttnn.permute(mask, (2, 0, 1))
            mask = ttnn.reshape(mask, (1, K, 1, -1))
            mask = (-1 * mask + 1) * -1e9
            self.bias = ttnn.add(self.bias, mask)
        x = self._to_torch(
            self.module(
                self._from_torch(a),
                self._from_torch(s),
                self.bias,
                self.keys_indexing,
            )
        )
        return x


class MSAModule(TorchWrapper):
    def __init__(
        self,
        n_blocks: int,
        avg_head_dim: int,
        avg_n_heads: int,
        tri_att_head_dim: int,
        tri_att_n_heads: int,
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.avg_head_dim = avg_head_dim
        self.avg_n_heads = avg_n_heads
        self.tri_att_head_dim = tri_att_head_dim
        self.tri_att_n_heads = tri_att_n_heads

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        self.module = MSA(
            self.n_blocks,
            self.avg_head_dim,
            self.avg_n_heads,
            self.tri_att_head_dim,
            self.tri_att_n_heads,
            filter_dict(state_dict, prefix[:-1]),
            self.compute_kernel_config,
        )

    @timing_decorator
    def forward(
        self,
        z: torch.Tensor,
        emb: torch.Tensor,
        feats: Dict[str, torch.Tensor],
        use_kernels: bool = False,
    ) -> torch.Tensor:
        m = torch.cat(
            [
                torch.nn.functional.one_hot(feats["msa"], num_classes=33),
                feats["has_deletion"].unsqueeze(-1),
                feats["deletion_value"].unsqueeze(-1),
                feats["msa_paired"].unsqueeze(-1),
            ],
            dim=-1,
        )
        return self._to_torch(
            self.module(
                self._from_torch(z),
                self._from_torch(m),
                self._from_torch(emb),
            )
        )

