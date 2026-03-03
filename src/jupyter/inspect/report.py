"""Report generation for ONNX models.

Provides:
- ReportView: Comprehensive HTML report
- generate_report(): Create a full model report
"""

from __future__ import annotations

import html as _html
from datetime import datetime
from typing import Any, Dict


try:
    from IPython.display import HTML, display
except ImportError:
    HTML = None
    display = print

try:
    import onnx
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


class ReportView:
    """Comprehensive model report."""
    
    def __init__(
        self,
        model: Any,
        name: str = "",
        include_graph: bool = True,
        include_weights: bool = True,
        include_shapes: bool = True,
    ):
        self._model = model
        self.name = name or "Model Report"
        self.include_graph = include_graph
        self.include_weights = include_weights
        self.include_shapes = include_shapes
        
        # Extract model info
        self._info = self._extract_info()
    
    def _extract_info(self) -> Dict[str, Any]:
        """Extract all relevant model information."""
        model = self._model
        if hasattr(model, '_model'):
            model = model._model
        
        info = {
            "name": self.name,
            "generated_at": datetime.now().isoformat(),
            "opset": None,
            "producer": None,
            "domain": None,
            "ir_version": None,
            "nodes": [],
            "inputs": [],
            "outputs": [],
            "initializers": [],
            "metadata": {},
        }
        
        if not HAS_ONNX or not isinstance(model, onnx.ModelProto):
            return info
        
        # Basic info
        if model.opset_import:
            info["opset"] = model.opset_import[0].version
        info["producer"] = model.producer_name or "unknown"
        info["domain"] = model.domain or ""
        info["ir_version"] = model.ir_version
        
        # Graph info
        graph = model.graph
        
        # Nodes
        for node in graph.node:
            info["nodes"].append({
                "name": node.name or node.op_type,
                "op_type": node.op_type,
                "domain": node.domain or "",
                "inputs": list(node.input),
                "outputs": list(node.output),
            })
        
        # Inputs
        init_names = {i.name for i in graph.initializer}
        for inp in graph.input:
            if inp.name not in init_names:
                shape = []
                if inp.type.tensor_type.shape:
                    for dim in inp.type.tensor_type.shape.dim:
                        if dim.dim_value:
                            shape.append(dim.dim_value)
                        elif dim.dim_param:
                            shape.append(dim.dim_param)
                        else:
                            shape.append("?")
                
                dtype = onnx.TensorProto.DataType.Name(inp.type.tensor_type.elem_type)
                info["inputs"].append({
                    "name": inp.name,
                    "shape": shape,
                    "dtype": dtype,
                })
        
        # Outputs
        for out in graph.output:
            shape = []
            if out.type.tensor_type.shape:
                for dim in out.type.tensor_type.shape.dim:
                    if dim.dim_value:
                        shape.append(dim.dim_value)
                    elif dim.dim_param:
                        shape.append(dim.dim_param)
                    else:
                        shape.append("?")
            
            dtype = onnx.TensorProto.DataType.Name(out.type.tensor_type.elem_type)
            info["outputs"].append({
                "name": out.name,
                "shape": shape,
                "dtype": dtype,
            })
        
        # Initializers (parameters)
        for init in graph.initializer:
            try:
                arr = onnx.numpy_helper.to_array(init)
                info["initializers"].append({
                    "name": init.name,
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                    "size_bytes": arr.nbytes,
                    "numel": arr.size,
                })
            except Exception:
                pass
        
        # Metadata
        for prop in model.metadata_props:
            info["metadata"][prop.key] = prop.value
        
        return info
    
    def _make_summary_section(self) -> str:
        """Generate summary section HTML."""
        info = self._info
        
        n_nodes = len(info["nodes"])
        n_inputs = len(info["inputs"])
        n_outputs = len(info["outputs"])
        n_params = len(info["initializers"])
        total_params = sum(i.get("numel", 0) for i in info["initializers"])
        total_size = sum(i.get("size_bytes", 0) for i in info["initializers"])
        
        # Format size
        if total_size < 1024:
            size_str = f"{total_size} B"
        elif total_size < 1024 * 1024:
            size_str = f"{total_size / 1024:.1f} KB"
        elif total_size < 1024 * 1024 * 1024:
            size_str = f"{total_size / (1024 * 1024):.1f} MB"
        else:
            size_str = f"{total_size / (1024 * 1024 * 1024):.2f} GB"
        
        return f"""
        <section style="margin-bottom: 24px;">
            <h2 style="font-size: 18px; margin-bottom: 12px; border-bottom: 1px solid #eee; padding-bottom: 8px;">
                📋 Summary
            </h2>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 12px;">
                <div style="text-align: center; padding: 12px; background: #f8f9fa; border-radius: 8px;">
                    <div style="font-size: 24px; font-weight: 600; color: #667eea;">{n_nodes}</div>
                    <div style="font-size: 12px; color: #666;">Nodes</div>
                </div>
                <div style="text-align: center; padding: 12px; background: #f8f9fa; border-radius: 8px;">
                    <div style="font-size: 24px; font-weight: 600; color: #667eea;">{n_inputs}</div>
                    <div style="font-size: 12px; color: #666;">Inputs</div>
                </div>
                <div style="text-align: center; padding: 12px; background: #f8f9fa; border-radius: 8px;">
                    <div style="font-size: 24px; font-weight: 600; color: #667eea;">{n_outputs}</div>
                    <div style="font-size: 12px; color: #666;">Outputs</div>
                </div>
                <div style="text-align: center; padding: 12px; background: #f8f9fa; border-radius: 8px;">
                    <div style="font-size: 24px; font-weight: 600; color: #667eea;">{n_params}</div>
                    <div style="font-size: 12px; color: #666;">Parameters</div>
                </div>
                <div style="text-align: center; padding: 12px; background: #f8f9fa; border-radius: 8px;">
                    <div style="font-size: 24px; font-weight: 600; color: #667eea;">{total_params:,}</div>
                    <div style="font-size: 12px; color: #666;">Total Params</div>
                </div>
                <div style="text-align: center; padding: 12px; background: #f8f9fa; border-radius: 8px;">
                    <div style="font-size: 24px; font-weight: 600; color: #667eea;">{size_str}</div>
                    <div style="font-size: 12px; color: #666;">Model Size</div>
                </div>
            </div>
            <div style="margin-top: 12px; font-size: 12px; color: #666;">
                <strong>Opset:</strong> {info.get('opset', 'N/A')} | 
                <strong>Producer:</strong> {info.get('producer', 'N/A')} |
                <strong>IR Version:</strong> {info.get('ir_version', 'N/A')}
            </div>
        </section>
        """
    
    def _make_io_section(self) -> str:
        """Generate inputs/outputs section."""
        info = self._info
        
        # Inputs table
        input_rows = []
        for inp in info["inputs"]:
            shape_str = ", ".join(str(d) for d in inp["shape"])
            input_rows.append(f"""
            <tr>
                <td style="padding: 6px; font-size: 12px;">{_html.escape(inp['name'])}</td>
                <td style="padding: 6px; font-size: 12px;">[{shape_str}]</td>
                <td style="padding: 6px; font-size: 12px;">{inp['dtype']}</td>
            </tr>
            """)
        
        # Outputs table
        output_rows = []
        for out in info["outputs"]:
            shape_str = ", ".join(str(d) for d in out["shape"])
            output_rows.append(f"""
            <tr>
                <td style="padding: 6px; font-size: 12px;">{_html.escape(out['name'])}</td>
                <td style="padding: 6px; font-size: 12px;">[{shape_str}]</td>
                <td style="padding: 6px; font-size: 12px;">{out['dtype']}</td>
            </tr>
            """)
        
        return f"""
        <section style="margin-bottom: 24px;">
            <h2 style="font-size: 18px; margin-bottom: 12px; border-bottom: 1px solid #eee; padding-bottom: 8px;">
                📥 Inputs & Outputs
            </h2>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
                <div>
                    <h3 style="font-size: 14px; margin-bottom: 8px; color: #28a745;">Inputs ({len(info['inputs'])})</h3>
                    <table style="width: 100%; border-collapse: collapse; border: 1px solid #eee;">
                        <tr style="background: #f8f9fa;">
                            <th style="text-align: left; padding: 6px; font-size: 11px;">Name</th>
                            <th style="text-align: left; padding: 6px; font-size: 11px;">Shape</th>
                            <th style="text-align: left; padding: 6px; font-size: 11px;">Dtype</th>
                        </tr>
                        {"".join(input_rows)}
                    </table>
                </div>
                <div>
                    <h3 style="font-size: 14px; margin-bottom: 8px; color: #dc3545;">Outputs ({len(info['outputs'])})</h3>
                    <table style="width: 100%; border-collapse: collapse; border: 1px solid #eee;">
                        <tr style="background: #f8f9fa;">
                            <th style="text-align: left; padding: 6px; font-size: 11px;">Name</th>
                            <th style="text-align: left; padding: 6px; font-size: 11px;">Shape</th>
                            <th style="text-align: left; padding: 6px; font-size: 11px;">Dtype</th>
                        </tr>
                        {"".join(output_rows)}
                    </table>
                </div>
            </div>
        </section>
        """
    
    def _make_ops_section(self) -> str:
        """Generate op distribution section."""
        info = self._info
        
        # Count ops
        op_counts: Dict[str, int] = {}
        for node in info["nodes"]:
            op = node["op_type"]
            op_counts[op] = op_counts.get(op, 0) + 1
        
        sorted_ops = sorted(op_counts.items(), key=lambda x: -x[1])
        max_count = sorted_ops[0][1] if sorted_ops else 1
        
        bars = []
        for op, count in sorted_ops[:15]:
            pct = 100 * count / max_count
            bars.append(f"""
            <div style="display: flex; align-items: center; margin: 4px 0;">
                <div style="width: 120px; font-size: 12px; overflow: hidden; text-overflow: ellipsis;">{op}</div>
                <div style="flex: 1; background: #eee; border-radius: 4px; margin: 0 8px; height: 16px;">
                    <div style="width: {pct}%; background: #667eea; height: 100%; border-radius: 4px;"></div>
                </div>
                <div style="width: 40px; font-size: 12px; text-align: right;">{count}</div>
            </div>
            """)
        
        return f"""
        <section style="margin-bottom: 24px;">
            <h2 style="font-size: 18px; margin-bottom: 12px; border-bottom: 1px solid #eee; padding-bottom: 8px;">
                📊 Operator Distribution
            </h2>
            <div style="max-width: 500px;">
                {"".join(bars)}
            </div>
            {f'<div style="font-size: 11px; color: #999; margin-top: 8px;">Showing top 15 of {len(op_counts)} op types</div>' if len(op_counts) > 15 else ''}
        </section>
        """
    
    def _make_weights_section(self) -> str:
        """Generate weights summary section."""
        info = self._info
        
        if not self.include_weights or not info["initializers"]:
            return ""
        
        # Sort by size
        sorted_params = sorted(info["initializers"], key=lambda x: -x.get("size_bytes", 0))
        
        rows = []
        for p in sorted_params[:20]:
            shape_str = ", ".join(str(d) for d in p["shape"])
            size_bytes = p.get("size_bytes", 0)
            if size_bytes < 1024:
                size_str = f"{size_bytes} B"
            elif size_bytes < 1024 * 1024:
                size_str = f"{size_bytes / 1024:.1f} KB"
            else:
                size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
            
            rows.append(f"""
            <tr>
                <td style="padding: 6px; font-size: 11px; max-width: 200px; overflow: hidden; text-overflow: ellipsis;">
                    {_html.escape(p['name'][:50])}
                </td>
                <td style="padding: 6px; font-size: 11px;">[{shape_str}]</td>
                <td style="padding: 6px; font-size: 11px;">{p['dtype']}</td>
                <td style="padding: 6px; font-size: 11px; text-align: right;">{p.get('numel', 0):,}</td>
                <td style="padding: 6px; font-size: 11px; text-align: right;">{size_str}</td>
            </tr>
            """)
        
        return f"""
        <section style="margin-bottom: 24px;">
            <h2 style="font-size: 18px; margin-bottom: 12px; border-bottom: 1px solid #eee; padding-bottom: 8px;">
                ⚖️ Parameters (Top 20 by Size)
            </h2>
            <table style="width: 100%; border-collapse: collapse; border: 1px solid #eee;">
                <tr style="background: #f8f9fa;">
                    <th style="text-align: left; padding: 6px; font-size: 11px;">Name</th>
                    <th style="text-align: left; padding: 6px; font-size: 11px;">Shape</th>
                    <th style="text-align: left; padding: 6px; font-size: 11px;">Dtype</th>
                    <th style="text-align: right; padding: 6px; font-size: 11px;">Elements</th>
                    <th style="text-align: right; padding: 6px; font-size: 11px;">Size</th>
                </tr>
                {"".join(rows)}
            </table>
        </section>
        """
    
    def _make_graph_section(self) -> str:
        """Generate graph visualization section."""
        if not self.include_graph:
            return ""
        
        try:
            from .graph import GraphView
            graph_view = GraphView(self._model, name=self.name, max_nodes=100)
            svg = graph_view._render_to_svg()
            
            return f"""
            <section style="margin-bottom: 24px;">
                <h2 style="font-size: 18px; margin-bottom: 12px; border-bottom: 1px solid #eee; padding-bottom: 8px;">
                    📈 Graph
                </h2>
                <div style="overflow-x: auto; background: #f8f9fa; padding: 12px; border-radius: 8px;">
                    {svg}
                </div>
            </section>
            """
        except Exception as e:
            return f"""
            <section style="margin-bottom: 24px;">
                <h2 style="font-size: 18px; margin-bottom: 12px;">📈 Graph</h2>
                <div style="color: #999;">Graph visualization not available: {e}</div>
            </section>
            """
    
    def _make_metadata_section(self) -> str:
        """Generate metadata section."""
        info = self._info
        
        if not info["metadata"]:
            return ""
        
        rows = []
        for key, value in info["metadata"].items():
            # Truncate long values
            val_str = str(value)
            if len(val_str) > 100:
                val_str = val_str[:100] + "..."
            rows.append(f"""
            <tr>
                <td style="padding: 6px; font-size: 12px; font-weight: 500;">{_html.escape(key)}</td>
                <td style="padding: 6px; font-size: 12px;">{_html.escape(val_str)}</td>
            </tr>
            """)
        
        return f"""
        <section style="margin-bottom: 24px;">
            <h2 style="font-size: 18px; margin-bottom: 12px; border-bottom: 1px solid #eee; padding-bottom: 8px;">
                🏷️ Metadata
            </h2>
            <table style="width: 100%; border-collapse: collapse; border: 1px solid #eee;">
                {"".join(rows)}
            </table>
        </section>
        """
    
    def _repr_html_(self) -> str:
        return f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    max-width: 900px; margin: 0 auto; padding: 20px;">
            <header style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                          color: white; padding: 24px; border-radius: 12px; margin-bottom: 24px;">
                <h1 style="margin: 0; font-size: 24px;">📄 {_html.escape(self.name)}</h1>
                <div style="margin-top: 8px; font-size: 12px; opacity: 0.9;">
                    Generated: {self._info['generated_at']}
                </div>
            </header>
            
            {self._make_summary_section()}
            {self._make_io_section()}
            {self._make_ops_section()}
            {self._make_weights_section()}
            {self._make_graph_section()}
            {self._make_metadata_section()}
            
            <footer style="text-align: center; padding: 16px; color: #999; font-size: 11px;">
                Generated by Fuse Visual Toolkit
            </footer>
        </div>
        """
    
    def to_html(self) -> str:
        """Export full HTML document."""
        body = self._repr_html_()
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>{_html.escape(self.name)}</title>
            <style>
                body {{ margin: 0; padding: 20px; background: #f5f5f5; }}
                * {{ box-sizing: border-box; }}
            </style>
        </head>
        <body>
            {body}
        </body>
        </html>
        """
    
    def save(self, path: str) -> None:
        """Save report to HTML file."""
        with open(path, "w") as f:
            f.write(self.to_html())
    
    def to_json(self) -> Dict[str, Any]:
        """Export report data as JSON."""
        return self._info


def report(model: Any, name: str = "", **kwargs) -> ReportView:
    """Generate a comprehensive model report.
    
    Args:
        model: ONNX model or Model wrapper
        name: Report title
        **kwargs: Options passed to ReportView
    """
    return ReportView(model, name, **kwargs)


def export_report(model: Any, path: str, name: str = "", **kwargs) -> None:
    """Generate and save a model report to file.
    
    Args:
        model: ONNX model or Model wrapper
        path: Output file path (.html)
        name: Report title
        **kwargs: Options passed to ReportView
    """
    r = ReportView(model, name, **kwargs)
    r.save(path)
