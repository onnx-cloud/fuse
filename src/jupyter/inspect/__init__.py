"""Fuse Jupyter Visual Inspection Toolkit.

A modular library for inspecting tensors, graphs, and models in Jupyter notebooks.
Provides rich HTML/widget visualizations with multiple interaction paradigms:

1. Python API:      tensor.inspect(), tensor.as_image(), etc.
2. Magic commands:  %inspect x, %inspect x as image
3. Emoji shortcuts: x | 🖼️

Example:
    from src.jupyter.inspect import inspect, as_image, TensorProxy
    
    # Direct API
    inspect(my_tensor)
    as_image(my_tensor)
    
    # With proxy for chaining
    t = TensorProxy(my_tensor)
    t | 🖼️   # Display as image
    
    # Magic commands (in IPython/Jupyter)
    %load_ext src.jupyter.inspect.magics
    %inspect x as image
    %stats x
    %compare a b
    
    # Graph & Model inspection
    %graph model
    %trace model
    %weights model
    %report model
"""

from .core import TensorView, TensorInspector, TensorProxy, inspect
from .registry import register_decoder, get_decoder, list_decoders
from .decoders import (
    # Decoder classes
    ImageView,
    AudioView,
    TokenView,
    AttentionView,
    EmbeddingView,
    VideoView,
    PointCloudView,
    BoxView,
    # Decoder functions
    as_image,
    as_audio,
    as_tokens,
    as_attention,
    as_embeddings,
    as_video,
    as_points,
    as_boxes,
)
from .analysis import (
    # Dimensionality reduction
    pca,
    tsne,
    umap_project,
    # Distribution analysis
    histogram,
    HistogramView,
    # Comparison
    compare_tensors,
    ComparisonView,
    # Statistics
    describe,
    DescribeView,
    # Outliers
    detect_outliers,
    # Correlation
    correlation_matrix,
    CorrelationView,
)

# Graph visualization
from .graph import (
    GraphView,
    GraphDiff,
    GraphStats,
    graph,
    graph_diff,
    graph_stats,
)

# Execution tracing
from .trace import (
    TraceView,
    TraceResult,
    trace,
)

# Weight inspection
from .weights import (
    WeightsView,
    FilterView,
    weights,
    filters,
    sparsity_analysis,
    magnitude_analysis,
)

# Tensor slicing
from .slice import (
    SliceView,
    slice_tensor,
    head,
    tail,
    sample,
)

# Report generation
from .report import (
    ReportView,
    report,
    export_report,
)

# Auto-register magics if in IPython
try:
    from .magics import register_magics, InspectMagics
    register_magics()
except Exception:
    pass

__all__ = [
    # Core
    "TensorView",
    "TensorInspector",
    "TensorProxy",
    "inspect",
    # Registry
    "register_decoder",
    "get_decoder",
    "list_decoders",
    # Decoder views
    "ImageView",
    "AudioView",
    "TokenView",
    "AttentionView",
    "EmbeddingView",
    "VideoView",
    "PointCloudView",
    "BoxView",
    # Decoder functions
    "as_image",
    "as_audio", 
    "as_tokens",
    "as_attention",
    "as_embeddings",
    "as_video",
    "as_points",
    "as_boxes",
    # Analysis - Dimensionality reduction
    "pca",
    "tsne",
    "umap_project",
    # Analysis - Distribution
    "histogram",
    "HistogramView",
    # Analysis - Comparison
    "compare_tensors",
    "ComparisonView",
    # Analysis - Statistics
    "describe",
    "DescribeView",
    # Analysis - Outliers
    "detect_outliers",
    # Analysis - Correlation
    "correlation_matrix",
    "CorrelationView",
    # Graph visualization
    "GraphView",
    "GraphDiff",
    "GraphStats",
    "graph",
    "graph_diff",
    "graph_stats",
    # Execution tracing
    "TraceView",
    "TraceResult",
    "trace",
    # Weight inspection
    "WeightsView",
    "FilterView",
    "weights",
    "filters",
    "sparsity_analysis",
    "magnitude_analysis",
    # Tensor slicing
    "SliceView",
    "slice_tensor",
    "head",
    "tail",
    "sample",
    # Report generation
    "ReportView",
    "report",
    "export_report",
    # Magics
    "InspectMagics",
    "register_magics",
]
