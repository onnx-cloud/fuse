"""Execution tracing and profiling for ONNX models.

Provides:
- TraceView: Visualize execution timing
- trace(): Run model with profiling
"""

from __future__ import annotations

import html as _html
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from IPython.display import HTML, display
except ImportError:
    HTML = None
    display = print

try:
    import onnx
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False


@dataclass
class NodeTiming:
    """Timing information for a single node."""
    name: str
    op_type: str
    duration_us: float  # microseconds
    
    @property
    def duration_ms(self) -> float:
        return self.duration_us / 1000


@dataclass
class TraceResult:
    """Results from model execution trace."""
    model_name: str
    total_duration_ms: float
    node_timings: List[NodeTiming]
    input_shapes: Dict[str, Tuple[int, ...]]
    output_shapes: Dict[str, Tuple[int, ...]]
    device: str = "cpu"
    
    @property
    def n_nodes(self) -> int:
        return len(self.node_timings)
    
    def top_nodes(self, n: int = 10) -> List[NodeTiming]:
        """Get top N slowest nodes."""
        return sorted(self.node_timings, key=lambda x: -x.duration_us)[:n]
    
    def by_op_type(self) -> Dict[str, float]:
        """Aggregate timing by op type."""
        by_op: Dict[str, float] = {}
        for t in self.node_timings:
            by_op[t.op_type] = by_op.get(t.op_type, 0) + t.duration_us
        return dict(sorted(by_op.items(), key=lambda x: -x[1]))


class TraceView:
    """Visualization of execution trace."""
    
    def __init__(self, result: TraceResult):
        self.result = result
    
    def _make_timeline_svg(self, width: int = 600, height: int = 200) -> str:
        """Generate timeline SVG."""
        result = self.result
        if not result.node_timings:
            return "<em>No timing data</em>"
        
        # Compute cumulative times
        timings = result.node_timings
        total = sum(t.duration_us for t in timings)
        if total == 0:
            return "<em>No timing data</em>"
        
        margin = {"left": 40, "right": 10, "top": 30, "bottom": 20}
        plot_w = width - margin["left"] - margin["right"]
        plot_h = height - margin["top"] - margin["bottom"]
        
        # Generate bars
        bars = []
        x = margin["left"]
        bar_height = min(20, plot_h / len(timings))
        
        # Color by op type
        op_colors = {
            "Conv": "#e74c3c", "MatMul": "#e74c3c", "Gemm": "#e74c3c",
            "Relu": "#2ecc71", "Softmax": "#2ecc71",
            "BatchNormalization": "#9b59b6", "LayerNormalization": "#9b59b6",
            "Add": "#1abc9c", "Mul": "#1abc9c",
        }
        default_color = "#3498db"
        
        for i, t in enumerate(timings[:50]):  # Max 50 bars
            w = (t.duration_us / total) * plot_w
            if w < 1:
                w = 1
            y = margin["top"] + (i * bar_height)
            color = op_colors.get(t.op_type, default_color)
            
            bars.append(f"""
            <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{bar_height - 1:.1f}" 
                  fill="{color}" rx="2">
                <title>{t.name}: {t.duration_ms:.3f}ms ({t.op_type})</title>
            </rect>
            """)
            x += w
        
        # Time axis
        time_labels = []
        for pct in [0, 25, 50, 75, 100]:
            x_pos = margin["left"] + (pct / 100) * plot_w
            time_ms = (pct / 100) * result.total_duration_ms
            time_labels.append(
                f'<text x="{x_pos:.1f}" y="{height - 5}" font-size="10" '
                f'text-anchor="middle" fill="#666">{time_ms:.1f}ms</text>'
            )
        
        return f"""
        <svg width="{width}" height="{height}" style="background: #fafafa; border-radius: 4px;">
            <text x="{width/2}" y="15" font-size="12" text-anchor="middle" fill="#333" font-weight="600">
                Execution Timeline
            </text>
            {"".join(bars)}
            {"".join(time_labels)}
        </svg>
        """
    
    def _make_top_nodes_html(self) -> str:
        """Generate HTML for top slowest nodes."""
        top = self.result.top_nodes(10)
        total = self.result.total_duration_ms
        
        rows = []
        for t in top:
            pct = 100 * t.duration_ms / total if total > 0 else 0
            bar_width = min(100, pct)
            rows.append(f"""
            <tr>
                <td style="font-size: 11px; max-width: 120px; overflow: hidden; text-overflow: ellipsis;">
                    {_html.escape(t.name[:30])}
                </td>
                <td style="font-size: 11px;">{t.op_type}</td>
                <td style="font-size: 11px; text-align: right;">{t.duration_ms:.3f}ms</td>
                <td style="font-size: 11px; text-align: right;">{pct:.1f}%</td>
                <td style="width: 100px;">
                    <div style="background: #eee; border-radius: 2px; height: 8px;">
                        <div style="background: #e74c3c; width: {bar_width}%; height: 100%; border-radius: 2px;"></div>
                    </div>
                </td>
            </tr>
            """)
        
        return f"""
        <table style="width: 100%; border-collapse: collapse; margin-top: 8px;">
            <tr style="background: #f8f9fa; font-size: 11px; font-weight: 600;">
                <th style="text-align: left; padding: 4px;">Node</th>
                <th style="text-align: left; padding: 4px;">Op</th>
                <th style="text-align: right; padding: 4px;">Time</th>
                <th style="text-align: right; padding: 4px;">%</th>
                <th></th>
            </tr>
            {"".join(rows)}
        </table>
        """
    
    def _make_by_op_html(self) -> str:
        """Generate HTML for time by op type."""
        by_op = self.result.by_op_type()
        total_us = sum(by_op.values())
        
        items = []
        for op, us in list(by_op.items())[:10]:
            pct = 100 * us / total_us if total_us > 0 else 0
            ms = us / 1000
            items.append(f"""
            <div style="display: flex; align-items: center; margin: 2px 0;">
                <div style="width: 100px; font-size: 11px;">{op}</div>
                <div style="flex: 1; background: #eee; border-radius: 2px; height: 10px; margin: 0 8px;">
                    <div style="background: #3498db; width: {pct}%; height: 100%; border-radius: 2px;"></div>
                </div>
                <div style="width: 60px; font-size: 10px; text-align: right;">{ms:.2f}ms</div>
            </div>
            """)
        
        return f"""
        <div style="margin-top: 8px;">
            <div style="font-weight: 600; font-size: 12px; margin-bottom: 4px;">Time by Op Type</div>
            {"".join(items)}
        </div>
        """
    
    def _repr_html_(self) -> str:
        r = self.result
        timeline = self._make_timeline_svg()
        top_nodes = self._make_top_nodes_html()
        by_op = self._make_by_op_html()
        
        return f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    border: 1px solid #e1e4e8; border-radius: 8px; margin: 8px 0;">
            <div style="background: linear-gradient(135deg, #f39c12 0%, #e74c3c 100%);
                        color: white; padding: 12px; border-radius: 8px 8px 0 0;">
                <span style="font-weight: 600; font-size: 16px;">⏱️ Execution Trace: {_html.escape(r.model_name)}</span>
                <span style="float: right; font-size: 13px;">
                    {r.total_duration_ms:.2f}ms total
                </span>
            </div>
            
            <div style="padding: 12px;">
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-bottom: 12px;">
                    <div style="text-align: center; padding: 6px; background: #f8f9fa; border-radius: 4px;">
                        <div style="font-size: 16px; font-weight: 600;">{r.total_duration_ms:.2f}ms</div>
                        <div style="font-size: 10px; color: #666;">Total Time</div>
                    </div>
                    <div style="text-align: center; padding: 6px; background: #f8f9fa; border-radius: 4px;">
                        <div style="font-size: 16px; font-weight: 600;">{r.n_nodes}</div>
                        <div style="font-size: 10px; color: #666;">Nodes</div>
                    </div>
                    <div style="text-align: center; padding: 6px; background: #f8f9fa; border-radius: 4px;">
                        <div style="font-size: 16px; font-weight: 600;">{len(r.input_shapes)}</div>
                        <div style="font-size: 10px; color: #666;">Inputs</div>
                    </div>
                    <div style="text-align: center; padding: 6px; background: #f8f9fa; border-radius: 4px;">
                        <div style="font-size: 16px; font-weight: 600;">{r.device}</div>
                        <div style="font-size: 10px; color: #666;">Device</div>
                    </div>
                </div>
                
                {timeline}
                
                <details open>
                    <summary style="cursor: pointer; font-weight: 600; margin-top: 12px;">Top 10 Slowest Nodes</summary>
                    {top_nodes}
                </details>
                
                {by_op}
            </div>
        </div>
        """


def trace(
    model: Any,
    inputs: Optional[Dict[str, np.ndarray]] = None,
    name: str = "",
    device: str = "cpu",
    warmup_runs: int = 1,
    profile_runs: int = 3,
) -> TraceView:
    """Run model with execution profiling.
    
    Args:
        model: ONNX model (path, bytes, or ModelProto)
        inputs: Input tensors (auto-generated if None)
        name: Display name
        device: 'cpu' or 'cuda'
        warmup_runs: Number of warmup iterations
        profile_runs: Number of profiling iterations
        
    Returns:
        TraceView with timing information
    """
    if not HAS_ORT:
        raise ImportError("onnxruntime is required for tracing")
    
    # Load model
    if isinstance(model, str):
        model_proto = onnx.load(model)
        model_name = name or model
    elif isinstance(model, bytes):
        model_proto = onnx.load_from_string(model)
        model_name = name or "model"
    elif hasattr(model, '_model'):
        model_proto = model._model
        model_name = name or getattr(model, 'name', 'model')
    else:
        model_proto = model
        model_name = name or "model"
    
    # Create session with profiling
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
    sess_options = ort.SessionOptions()
    sess_options.enable_profiling = True
    
    session = ort.InferenceSession(
        model_proto.SerializeToString(),
        sess_options,
        providers=providers,
    )
    
    # Auto-generate inputs if not provided
    if inputs is None:
        inputs = {}
        for inp in session.get_inputs():
            shape = [d if isinstance(d, int) else 1 for d in inp.shape]
            dtype = np.float32
            if 'float16' in inp.type:
                dtype = np.float16
            elif 'int64' in inp.type:
                dtype = np.int64
            elif 'int32' in inp.type:
                dtype = np.int32
            inputs[inp.name] = np.random.randn(*shape).astype(dtype)
    
    input_shapes = {k: v.shape for k, v in inputs.items()}
    
    # Warmup
    for _ in range(warmup_runs):
        session.run(None, inputs)
    
    # Profile runs
    timings = []
    for _ in range(profile_runs):
        start = time.perf_counter()
        outputs = session.run(None, inputs)
        end = time.perf_counter()
        timings.append((end - start) * 1000)  # ms
    
    total_duration = sum(timings) / len(timings)
    
    # Get output shapes
    output_shapes = {}
    output_names = [o.name for o in session.get_outputs()]
    for name_out, arr in zip(output_names, outputs):
        output_shapes[name_out] = arr.shape
    
    # Try to get node-level profiling from ORT
    node_timings = []
    try:
        profile_file = session.end_profiling()
        import json
        with open(profile_file) as f:
            profile_data = json.load(f)
        
        for event in profile_data:
            if event.get('cat') == 'Node':
                node_timings.append(NodeTiming(
                    name=event.get('name', 'unknown'),
                    op_type=event.get('args', {}).get('op_name', 'unknown'),
                    duration_us=event.get('dur', 0),
                ))
        
        # Clean up profile file
        import os
        os.remove(profile_file)
    except Exception:
        # Fallback: estimate based on op types
        for node in model_proto.graph.node:
            node_timings.append(NodeTiming(
                name=node.name or node.op_type,
                op_type=node.op_type,
                duration_us=total_duration * 1000 / len(model_proto.graph.node),
            ))
    
    result = TraceResult(
        model_name=model_name,
        total_duration_ms=total_duration,
        node_timings=node_timings,
        input_shapes=input_shapes,
        output_shapes=output_shapes,
        device=device,
    )
    
    return TraceView(result)
