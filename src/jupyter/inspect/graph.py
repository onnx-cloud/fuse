"""Graph visualization and analysis tools.

Provides:
- GraphView: Interactive ONNX graph visualization
- GraphDiff: Compare two graphs
- GraphStats: Graph statistics and metrics
"""

from __future__ import annotations

import html as _html
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Union

try:
    from IPython.display import HTML, display
except ImportError:
    HTML = None
    display = print

try:
    import onnx
    from onnx import ModelProto, GraphProto, NodeProto
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    ModelProto = Any
    GraphProto = Any
    NodeProto = Any


def _get_graph(model_or_graph: Union[ModelProto, GraphProto, Any]) -> Optional[GraphProto]:
    """Extract GraphProto from various inputs."""
    if not HAS_ONNX:
        return None
    if isinstance(model_or_graph, onnx.ModelProto):
        return model_or_graph.graph
    if isinstance(model_or_graph, onnx.GraphProto):
        return model_or_graph
    if hasattr(model_or_graph, 'graph'):
        return model_or_graph.graph
    if hasattr(model_or_graph, '_model') and hasattr(model_or_graph._model, 'graph'):
        return model_or_graph._model.graph
    return None


@dataclass
class NodeInfo:
    """Information about a graph node."""
    name: str
    op_type: str
    inputs: List[str]
    outputs: List[str]
    attributes: Dict[str, Any]
    domain: str = ""
    
    @classmethod
    def from_node(cls, node: NodeProto) -> "NodeInfo":
        attrs = {}
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.FLOAT:
                attrs[attr.name] = attr.f
            elif attr.type == onnx.AttributeProto.INT:
                attrs[attr.name] = attr.i
            elif attr.type == onnx.AttributeProto.STRING:
                attrs[attr.name] = attr.s.decode() if isinstance(attr.s, bytes) else attr.s
            elif attr.type == onnx.AttributeProto.FLOATS:
                attrs[attr.name] = list(attr.floats)
            elif attr.type == onnx.AttributeProto.INTS:
                attrs[attr.name] = list(attr.ints)
            else:
                attrs[attr.name] = f"<{onnx.AttributeProto.AttributeType.Name(attr.type)}>"
        
        return cls(
            name=node.name or f"{node.op_type}_{id(node)}",
            op_type=node.op_type,
            inputs=list(node.input),
            outputs=list(node.output),
            attributes=attrs,
            domain=node.domain or "",
        )


@dataclass 
class GraphInfo:
    """Extracted information about a graph."""
    name: str
    nodes: List[NodeInfo]
    inputs: List[str]
    outputs: List[str]
    initializers: Set[str]
    
    @classmethod
    def from_graph(cls, graph: GraphProto, name: str = "") -> "GraphInfo":
        nodes = [NodeInfo.from_node(n) for n in graph.node]
        inputs = [i.name for i in graph.input]
        outputs = [o.name for o in graph.output]
        initializers = {i.name for i in graph.initializer}
        
        return cls(
            name=name or graph.name or "graph",
            nodes=nodes,
            inputs=inputs,
            outputs=outputs,
            initializers=initializers,
        )
    
    @property
    def n_nodes(self) -> int:
        return len(self.nodes)
    
    @property
    def n_params(self) -> int:
        return len(self.initializers)
    
    def op_counts(self) -> Dict[str, int]:
        """Count occurrences of each op type."""
        counts: Dict[str, int] = {}
        for node in self.nodes:
            counts[node.op_type] = counts.get(node.op_type, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))


class GraphView:
    """Interactive graph visualization."""
    
    # Op type colors
    OP_COLORS = {
        # Compute ops
        "Conv": "#e74c3c",
        "MatMul": "#e74c3c", 
        "Gemm": "#e74c3c",
        # Activations
        "Relu": "#2ecc71",
        "Sigmoid": "#2ecc71",
        "Tanh": "#2ecc71",
        "Softmax": "#2ecc71",
        "Gelu": "#2ecc71",
        # Normalization
        "BatchNormalization": "#9b59b6",
        "LayerNormalization": "#9b59b6",
        "InstanceNormalization": "#9b59b6",
        # Pooling
        "MaxPool": "#3498db",
        "AveragePool": "#3498db",
        "GlobalAveragePool": "#3498db",
        # Shape ops
        "Reshape": "#f39c12",
        "Transpose": "#f39c12",
        "Squeeze": "#f39c12",
        "Unsqueeze": "#f39c12",
        "Flatten": "#f39c12",
        # Element-wise
        "Add": "#1abc9c",
        "Sub": "#1abc9c",
        "Mul": "#1abc9c",
        "Div": "#1abc9c",
        # Attention
        "Attention": "#e91e63",
        "MultiHeadAttention": "#e91e63",
    }
    DEFAULT_COLOR = "#95a5a6"
    
    def __init__(
        self,
        model_or_graph: Any,
        name: Optional[str] = None,
        layout: str = "TB",  # TB, LR, BT, RL
        show_shapes: bool = True,
        show_attrs: bool = False,
        max_nodes: int = 200,
        collapse_patterns: bool = True,
    ):
        self._graph = _get_graph(model_or_graph)
        if self._graph is None:
            raise ValueError("Could not extract graph from input")
        
        self._info = GraphInfo.from_graph(self._graph, name or "")
        self.name = name or self._info.name
        self.layout = layout
        self.show_shapes = show_shapes
        self.show_attrs = show_attrs
        self.max_nodes = max_nodes
        self.collapse_patterns = collapse_patterns
    
    def _get_op_color(self, op_type: str) -> str:
        return self.OP_COLORS.get(op_type, self.DEFAULT_COLOR)
    
    def _make_dot(self) -> str:
        """Generate DOT representation of the graph."""
        lines = [
            f'digraph "{_html.escape(self.name)}" {{',
            f'    rankdir={self.layout};',
            '    node [shape=box, style="rounded,filled", fontname="Arial", fontsize=10];',
            '    edge [fontname="Arial", fontsize=8];',
        ]
        
        info = self._info
        
        # Add input nodes
        for inp in info.inputs:
            if inp not in info.initializers:
                lines.append(f'    "{inp}" [shape=ellipse, fillcolor="#dfe6e9", label="{inp}"];')
        
        # Add nodes (truncate if too many)
        nodes_to_show = info.nodes[:self.max_nodes]
        for node in nodes_to_show:
            color = self._get_op_color(node.op_type)
            label = node.op_type
            if node.name and node.name != node.op_type:
                label = f"{node.op_type}\\n{node.name[:20]}"
            
            lines.append(f'    "{node.name}" [fillcolor="{color}", fontcolor="white", label="{label}"];')
        
        if len(info.nodes) > self.max_nodes:
            lines.append(f'    "..." [shape=plaintext, label="... +{len(info.nodes) - self.max_nodes} more nodes"];')
        
        # Add output nodes
        for out in info.outputs:
            lines.append(f'    "{out}" [shape=ellipse, fillcolor="#74b9ff", label="{out}"];')
        
        # Add edges
        node_outputs: Dict[str, str] = {}
        for node in nodes_to_show:
            for out in node.outputs:
                node_outputs[out] = node.name
        
        for node in nodes_to_show:
            for inp in node.inputs:
                if inp in node_outputs:
                    lines.append(f'    "{node_outputs[inp]}" -> "{node.name}";')
                elif inp in info.inputs and inp not in info.initializers:
                    lines.append(f'    "{inp}" -> "{node.name}";')
        
        # Connect to outputs
        for out in info.outputs:
            if out in node_outputs:
                lines.append(f'    "{node_outputs[out]}" -> "{out}";')
        
        lines.append("}")
        return "\n".join(lines)
    
    def _render_to_svg(self) -> str:
        """Render DOT to SVG using graphviz."""
        dot = self._make_dot()
        
        try:
            import subprocess
            result = subprocess.run(
                ["dot", "-Tsvg"],
                input=dot.encode(),
                capture_output=True,
                timeout=30,
            )
            if result.returncode == 0:
                svg = result.stdout.decode()
                # Extract just the SVG content
                start = svg.find("<svg")
                if start >= 0:
                    return svg[start:]
            return f"<pre>Error rendering graph: {result.stderr.decode()}</pre>"
        except FileNotFoundError:
            return "<pre>Graphviz 'dot' not found. Install graphviz.</pre>"
        except Exception as e:
            return f"<pre>Error: {e}</pre>"
    
    def _make_stats_html(self) -> str:
        """Generate stats panel HTML."""
        info = self._info
        op_counts = info.op_counts()
        
        # Top 10 ops
        top_ops = list(op_counts.items())[:10]
        ops_html = "".join(
            f'<div style="display: flex; justify-content: space-between; padding: 2px 0;">'
            f'<span style="color: {self._get_op_color(op)};">●</span> {op}: <strong>{count}</strong></div>'
            for op, count in top_ops
        )
        
        return f"""
        <div style="font-size: 12px; padding: 8px; background: #f8f9fa; border-radius: 4px;">
            <div style="font-weight: 600; margin-bottom: 8px;">📊 Statistics</div>
            <div>Nodes: <strong>{info.n_nodes}</strong></div>
            <div>Inputs: <strong>{len(info.inputs) - info.n_params}</strong></div>
            <div>Outputs: <strong>{len(info.outputs)}</strong></div>
            <div>Parameters: <strong>{info.n_params}</strong></div>
            <div style="margin-top: 8px; font-weight: 600;">Op Types:</div>
            {ops_html}
        </div>
        """
    
    def _repr_html_(self) -> str:
        svg = self._render_to_svg()
        stats = self._make_stats_html()
        info = self._info
        
        return f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    border: 1px solid #e1e4e8; border-radius: 8px; margin: 8px 0;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white; padding: 12px; border-radius: 8px 8px 0 0;">
                <span style="font-weight: 600; font-size: 16px;">📊 {_html.escape(self.name)}</span>
                <span style="float: right; font-size: 12px; opacity: 0.9;">
                    {info.n_nodes} nodes
                </span>
            </div>
            <div style="display: flex; flex-wrap: wrap;">
                <div style="flex: 1; min-width: 300px; padding: 12px; overflow-x: auto;">
                    {svg}
                </div>
                <div style="width: 200px; padding: 12px; border-left: 1px solid #e1e4e8;">
                    {stats}
                </div>
            </div>
        </div>
        """
    
    def to_svg(self) -> str:
        """Export as SVG string."""
        return self._render_to_svg()
    
    def to_dot(self) -> str:
        """Export as DOT string."""
        return self._make_dot()


class GraphDiff:
    """Compare two ONNX graphs."""
    
    def __init__(
        self,
        graph_a: Any,
        graph_b: Any,
        name_a: str = "A",
        name_b: str = "B",
    ):
        self._graph_a = _get_graph(graph_a)
        self._graph_b = _get_graph(graph_b)
        
        if self._graph_a is None or self._graph_b is None:
            raise ValueError("Could not extract graphs from inputs")
        
        self._info_a = GraphInfo.from_graph(self._graph_a, name_a)
        self._info_b = GraphInfo.from_graph(self._graph_b, name_b)
        self.name_a = name_a
        self.name_b = name_b
    
    def _compute_diff(self) -> Dict[str, Any]:
        """Compute differences between graphs."""
        nodes_a = {n.name: n for n in self._info_a.nodes}
        nodes_b = {n.name: n for n in self._info_b.nodes}
        
        names_a = set(nodes_a.keys())
        names_b = set(nodes_b.keys())
        
        added = names_b - names_a
        removed = names_a - names_b
        common = names_a & names_b
        
        # Check for modified nodes (same name, different op or attrs)
        modified = set()
        for name in common:
            na, nb = nodes_a[name], nodes_b[name]
            if na.op_type != nb.op_type or na.attributes != nb.attributes:
                modified.add(name)
        
        unchanged = common - modified
        
        # Op type changes
        ops_a = self._info_a.op_counts()
        ops_b = self._info_b.op_counts()
        
        return {
            "added": added,
            "removed": removed,
            "modified": modified,
            "unchanged": unchanged,
            "nodes_a": nodes_a,
            "nodes_b": nodes_b,
            "ops_a": ops_a,
            "ops_b": ops_b,
        }
    
    def _repr_html_(self) -> str:
        diff = self._compute_diff()
        
        n_added = len(diff["added"])
        n_removed = len(diff["removed"])
        n_modified = len(diff["modified"])
        n_unchanged = len(diff["unchanged"])
        
        # Summary
        summary_color = "#28a745" if n_added == 0 and n_removed == 0 and n_modified == 0 else "#dc3545"
        summary_text = "Identical" if summary_color == "#28a745" else "Different"
        
        # Op comparison
        all_ops = set(diff["ops_a"].keys()) | set(diff["ops_b"].keys())
        op_rows = []
        for op in sorted(all_ops):
            count_a = diff["ops_a"].get(op, 0)
            count_b = diff["ops_b"].get(op, 0)
            delta = count_b - count_a
            delta_str = f"+{delta}" if delta > 0 else str(delta) if delta < 0 else "="
            delta_color = "#28a745" if delta > 0 else "#dc3545" if delta < 0 else "#999"
            op_rows.append(
                f'<tr><td>{op}</td><td>{count_a}</td><td>{count_b}</td>'
                f'<td style="color: {delta_color}; font-weight: 600;">{delta_str}</td></tr>'
            )
        
        return f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    border: 1px solid #e1e4e8; border-radius: 8px; padding: 12px; margin: 8px 0;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <span style="font-weight: 600; font-size: 14px;">
                    🔀 Graph Diff: {_html.escape(self.name_a)} → {_html.escape(self.name_b)}
                </span>
                <span style="color: {summary_color}; font-weight: 600;">{summary_text}</span>
            </div>
            
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-bottom: 12px;">
                <div style="text-align: center; padding: 8px; background: #d4edda; border-radius: 4px;">
                    <div style="font-size: 20px; font-weight: 600; color: #28a745;">{n_added}</div>
                    <div style="font-size: 11px; color: #155724;">Added</div>
                </div>
                <div style="text-align: center; padding: 8px; background: #f8d7da; border-radius: 4px;">
                    <div style="font-size: 20px; font-weight: 600; color: #dc3545;">{n_removed}</div>
                    <div style="font-size: 11px; color: #721c24;">Removed</div>
                </div>
                <div style="text-align: center; padding: 8px; background: #fff3cd; border-radius: 4px;">
                    <div style="font-size: 20px; font-weight: 600; color: #856404;">{n_modified}</div>
                    <div style="font-size: 11px; color: #856404;">Modified</div>
                </div>
                <div style="text-align: center; padding: 8px; background: #e2e3e5; border-radius: 4px;">
                    <div style="font-size: 20px; font-weight: 600; color: #383d41;">{n_unchanged}</div>
                    <div style="font-size: 11px; color: #383d41;">Unchanged</div>
                </div>
            </div>
            
            <details>
                <summary style="cursor: pointer; font-weight: 600; margin-bottom: 8px;">Op Type Changes</summary>
                <table style="width: 100%; font-size: 12px; border-collapse: collapse;">
                    <tr style="background: #f8f9fa;">
                        <th style="text-align: left; padding: 4px;">Op</th>
                        <th>{_html.escape(self.name_a)}</th>
                        <th>{_html.escape(self.name_b)}</th>
                        <th>Δ</th>
                    </tr>
                    {"".join(op_rows)}
                </table>
            </details>
        </div>
        """


@dataclass
class GraphStats:
    """Comprehensive graph statistics."""
    
    name: str
    n_nodes: int
    n_inputs: int
    n_outputs: int
    n_params: int
    op_counts: Dict[str, int]
    estimated_flops: Optional[int] = None
    estimated_params: Optional[int] = None
    critical_path_length: Optional[int] = None
    
    @classmethod
    def from_graph(cls, graph: Any, name: str = "") -> "GraphStats":
        g = _get_graph(graph)
        if g is None:
            raise ValueError("Could not extract graph")
        
        info = GraphInfo.from_graph(g, name)
        
        return cls(
            name=info.name,
            n_nodes=info.n_nodes,
            n_inputs=len(info.inputs) - len(info.initializers),
            n_outputs=len(info.outputs),
            n_params=len(info.initializers),
            op_counts=info.op_counts(),
        )
    
    def _repr_html_(self) -> str:
        # Op distribution as mini bar chart
        max_count = max(self.op_counts.values()) if self.op_counts else 1
        op_bars = []
        for op, count in list(self.op_counts.items())[:15]:
            pct = 100 * count / max_count
            color = GraphView.OP_COLORS.get(op, GraphView.DEFAULT_COLOR)
            op_bars.append(f"""
            <div style="display: flex; align-items: center; margin: 2px 0;">
                <div style="width: 100px; font-size: 11px; overflow: hidden; text-overflow: ellipsis;">{op}</div>
                <div style="flex: 1; background: #eee; border-radius: 2px; margin: 0 8px;">
                    <div style="width: {pct}%; background: {color}; height: 12px; border-radius: 2px;"></div>
                </div>
                <div style="width: 30px; font-size: 11px; text-align: right;">{count}</div>
            </div>
            """)
        
        return f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    border: 1px solid #e1e4e8; border-radius: 8px; padding: 12px; margin: 8px 0;
                    background: linear-gradient(135deg, #f8f9fa 0%, #fff 100%);">
            <div style="font-weight: 600; font-size: 14px; margin-bottom: 12px;">
                📈 Graph Statistics: {_html.escape(self.name)}
            </div>
            
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-bottom: 12px;">
                <div style="text-align: center; padding: 8px; background: #fff; border: 1px solid #eee; border-radius: 4px;">
                    <div style="font-size: 18px; font-weight: 600;">{self.n_nodes}</div>
                    <div style="font-size: 11px; color: #666;">Nodes</div>
                </div>
                <div style="text-align: center; padding: 8px; background: #fff; border: 1px solid #eee; border-radius: 4px;">
                    <div style="font-size: 18px; font-weight: 600;">{self.n_inputs}</div>
                    <div style="font-size: 11px; color: #666;">Inputs</div>
                </div>
                <div style="text-align: center; padding: 8px; background: #fff; border: 1px solid #eee; border-radius: 4px;">
                    <div style="font-size: 18px; font-weight: 600;">{self.n_outputs}</div>
                    <div style="font-size: 11px; color: #666;">Outputs</div>
                </div>
                <div style="text-align: center; padding: 8px; background: #fff; border: 1px solid #eee; border-radius: 4px;">
                    <div style="font-size: 18px; font-weight: 600;">{self.n_params}</div>
                    <div style="font-size: 11px; color: #666;">Parameters</div>
                </div>
            </div>
            
            <div style="font-weight: 600; font-size: 12px; margin-bottom: 8px;">Op Distribution</div>
            {"".join(op_bars)}
        </div>
        """


def graph(model_or_graph: Any, name: Optional[str] = None, **kwargs) -> GraphView:
    """Display an ONNX graph.
    
    Args:
        model_or_graph: ONNX ModelProto, GraphProto, or Model wrapper
        name: Optional display name
        **kwargs: Options passed to GraphView
    """
    return GraphView(model_or_graph, name, **kwargs)


def graph_diff(a: Any, b: Any, name_a: str = "A", name_b: str = "B") -> GraphDiff:
    """Compare two ONNX graphs.
    
    Args:
        a: First model/graph
        b: Second model/graph
        name_a: Name for first graph
        name_b: Name for second graph
    """
    return GraphDiff(a, b, name_a, name_b)


def graph_stats(model_or_graph: Any, name: str = "") -> GraphStats:
    """Get statistics for an ONNX graph.
    
    Args:
        model_or_graph: ONNX ModelProto, GraphProto, or Model wrapper
        name: Optional display name
    """
    return GraphStats.from_graph(model_or_graph, name)
