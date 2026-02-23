"""Graph qualification utilities.

Handles adding scope prefixes to nodes and values in ONNX graphs,
extracted from main lowering logic for better modularity.
"""

from typing import Optional

import onnx
from ..graph_context import GraphContext


class GraphQualifier:
    """Handles qualification (scoping) of ONNX graph nodes and values."""
    
    def __init__(self, ctx: GraphContext, scope: str):
        """Initialize graph qualifier.
        
        Args:
            ctx: GraphContext for accessing node information
            scope: Scope prefix to apply
        """
        self.ctx = ctx
        self.scope = scope
    
    def qualify_graph(self, graph: onnx.GraphProto) -> onnx.GraphProto:
        """Add scope prefix to all nodes and values in a graph.
        
        Args:
            graph: ONNX GraphProto to qualify
            
        Returns:
            Qualified GraphProto
        """
        # Create mapping of old names to qualified names
        name_map = {}
        
        # Qualify inputs (but not if they're graph inputs)
        for inp in graph.input:
            if inp.name not in name_map:
                name_map[inp.name] = f"{self.scope}_{inp.name}"
        
        # Qualify outputs
        for out in graph.output:
            if out.name not in name_map:
                name_map[out.name] = f"{self.scope}_{out.name}"
        
        # Qualify initializers
        for init in graph.initializer:
            if init.name not in name_map:
                name_map[init.name] = f"{self.scope}_{init.name}"
        
        # Qualify value_info
        for vi in graph.value_info:
            if vi.name not in name_map:
                name_map[vi.name] = f"{self.scope}_{vi.name}"
        
        # Qualify nodes and their inputs/outputs
        qualified_graph = onnx.GraphProto()
        qualified_graph.CopyFrom(graph)
        qualified_graph.ClearField("node")
        
        for node in graph.node:
            qualified_node = onnx.NodeProto()
            qualified_node.CopyFrom(node)
            
            # Qualify node name
            if node.name:
                qualified_node.name = f"{self.scope}_{node.name}"
            
            # Qualify inputs
            qualified_node.ClearField("input")
            for inp in node.input:
                qualified_name = name_map.get(inp, f"{self.scope}_{inp}")
                qualified_node.input.append(qualified_name)
                name_map[inp] = qualified_name
            
            # Qualify outputs
            qualified_node.ClearField("output")
            for out in node.output:
                qualified_name = name_map.get(out, f"{self.scope}_{out}")
                qualified_node.output.append(qualified_name)
                name_map[out] = qualified_name
            
            # Handle subgraphs recursively
            for attr in qualified_node.attribute:
                if attr.HasField("g"):
                    sub_qualifier = GraphQualifier(self.ctx, f"{self.scope}_{attr.name}")
                    qualified_subgraph = sub_qualifier.qualify_graph(attr.g)
                    attr.g.CopyFrom(qualified_subgraph)
            
            qualified_graph.node.append(qualified_node)
        
        # Update value_info with qualified names
        qualified_graph.ClearField("value_info")
        for vi in graph.value_info:
            qualified_vi = onnx.ValueInfoProto()
            qualified_vi.CopyFrom(vi)
            qualified_vi.name = name_map.get(vi.name, f"{self.scope}_{vi.name}")
            qualified_graph.value_info.append(qualified_vi)
        
        # Update initializers with qualified names
        qualified_graph.ClearField("initializer")
        for init in graph.initializer:
            qualified_init = onnx.TensorProto()
            qualified_init.CopyFrom(init)
            qualified_init.name = name_map.get(init.name, f"{self.scope}_{init.name}")
            qualified_graph.initializer.append(qualified_init)
        
        # Update inputs with qualified names
        qualified_graph.ClearField("input")
        for inp in graph.input:
            qualified_inp = onnx.ValueInfoProto()
            qualified_inp.CopyFrom(inp)
            qualified_inp.name = name_map.get(inp.name, f"{self.scope}_{inp.name}")
            qualified_graph.input.append(qualified_inp)
        
        # Update outputs with qualified names
        qualified_graph.ClearField("output")
        for out in graph.output:
            qualified_out = onnx.ValueInfoProto()
            qualified_out.CopyFrom(out)
            qualified_out.name = name_map.get(out.name, f"{self.scope}_{out.name}")
            qualified_graph.output.append(qualified_out)
        
        return qualified_graph
    
    def qualify_node(self, node: onnx.NodeProto) -> onnx.NodeProto:
        """Qualify a single node.
        
        Args:
            node: Node to qualify
            
        Returns:
            Qualified node
        """
        qualified = onnx.NodeProto()
        qualified.CopyFrom(node)
        
        if node.name:
            qualified.name = f"{self.scope}_{node.name}"
        
        qualified.ClearField("input")
        for inp in node.input:
            qualified.input.append(f"{self.scope}_{inp}")
        
        qualified.ClearField("output")
        for out in node.output:
            qualified.output.append(f"{self.scope}_{out}")
        
        return qualified
