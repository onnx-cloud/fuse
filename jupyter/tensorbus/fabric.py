"""TensorFabric - Blackboard pattern for multi-model orchestration."""
from __future__ import annotations

import threading
from typing import Any, Callable, Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

import numpy as np

from .bus import TensorBus
from .core_types import Transport, SessionResolver, LocalSessionResolver, TensorBinding


logger = logging.getLogger(__name__)


class NodeState(Enum):
    """Execution state of a compute node."""
    IDLE = "idle"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ComputeNode:
    """A node in the computation graph representing a model or operation.
    
    Attributes:
        name: Unique node identifier
        model_path: Path to ONNX model (if applicable)
        graph_name: Graph name for ONNX model (extracted from metadata)
        inputs: List of input tensor names
        outputs: List of output tensor names
        bus: Optional dedicated TensorBus instance
        state: Current execution state
        dependencies: Set of node names that must complete first
        callback: Optional function to call on completion
        retries: Number of retry attempts on failure
    """
    name: str
    model_path: Optional[str] = None
    graph_name: Optional[str] = None
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    bus: Optional[TensorBus] = None
    state: NodeState = NodeState.IDLE
    dependencies: Set[str] = field(default_factory=set)
    callback: Optional[Callable[[ComputeNode], None]] = None
    retries: int = 0
    max_retries: int = 3
    error: Optional[str] = None


class TensorFabric:
    """Blackboard-based orchestrator for multi-model inference pipelines.
    
    TensorFabric implements the blackboard architectural pattern for
    coordinating multiple ONNX models and operations. It manages:
    
    - Computation graph with dependency tracking
    - Automatic execution scheduling based on data availability
    - Parallel execution of independent models
    - Error handling and retry logic
    - Centralized tensor sharing via TensorBus
    
    Architecture:
    - Blackboard: Central TensorBus for tensor storage/sharing
    - Compute Nodes: Individual models/operations with dependencies
    - Scheduler: Determines execution order and triggers nodes
    
    Example:
        fabric = TensorFabric()
        
        # Add models - inputs/outputs auto-extracted from ONNX metadata
        fabric.add_model("preprocess.onnx")
        fabric.add_model("model.onnx", name="inference", dependencies={"preprocess"})
        fabric.add_model("postprocess.onnx", dependencies={"inference"})
        
        # Execute pipeline (uses same contract as TensorBus)
        fabric.set_tensor("raw_image", image_data)  # Same as bus.set_tensor()
        fabric.execute()
        predictions = fabric.get_output("predictions")  # Convenience for numpy array
        
        # Or use get_tensor() for full TensorBinding (same as bus.get_tensor())
        binding = fabric.get_tensor("predictions")  # Returns TensorBinding
    
    Use Cases:
    - Multi-stage inference pipelines (preprocessing → model → postprocessing)
    - Ensemble models with voting/averaging
    - Heterogeneous model graphs (CPU + GPU nodes)
    - Dynamic execution based on intermediate results
    """

    def __init__(
        self,
        name: str = "fabric",
        resolver: Optional[SessionResolver] = None,
        transports: Optional[List[Transport]] = None,
        max_parallel: int = 4,
    ):
        """Initialize TensorFabric.
        
        Args:
            name: Fabric identifier
            resolver: Session resolver for ONNX models
            transports: List of transports for replication
            max_parallel: Maximum parallel node executions
        """
        self.name = name
        self.resolver = resolver or LocalSessionResolver(test_gpu=False)
        
        # Central blackboard (shared TensorBus)
        self.blackboard = TensorBus(
            name=f"{name}_blackboard",
            resolver=self.resolver,
            transports=transports or []
        )
        
        # Compute graph
        self._nodes: Dict[str, ComputeNode] = {}
        self._execution_order: List[str] = []
        
        # Execution control
        self._lock = threading.RLock()
        self._max_parallel = max_parallel
        self._active_workers: Set[str] = set()
        self._worker_threads: List[threading.Thread] = []
        
        # Completion tracking
        self._completed_nodes: Set[str] = set()
        self._failed_nodes: Set[str] = set()
    
    def add_model(
        self,
        model_path: str,
        name: Optional[str] = None,
        dependencies: Optional[Set[str]] = None,
        callback: Optional[Callable[[ComputeNode], None]] = None,
        dedicated_bus: bool = False,
    ) -> ComputeNode:
        """Add a model to the fabric.
        
        Automatically extracts inputs/outputs from the ONNX model metadata.
        
        Args:
            model_path: Path to ONNX model (required)
            name: Unique node name (defaults to graph name)
            dependencies: Set of node names that must complete first
            callback: Optional completion callback
            dedicated_bus: If True, create dedicated TensorBus for this node
        
        Returns:
            Created ComputeNode
        """
        with self._lock:
            # Create temporary bus to extract model metadata
            temp_bus = TensorBus(
                name=f"{self.name}_temp",
                resolver=self.resolver,
                transports=[]
            )
            binding = temp_bus.load_model(model_path)
            graph_name = binding.name
            
            # Extract inputs and outputs from session
            session = binding.session
            inputs = [arg.name for arg in session.get_inputs()]
            outputs = [arg.name for arg in session.get_outputs()]
            
            # Use graph name if no name provided
            node_name = name or graph_name
            
            if node_name in self._nodes:
                raise ValueError(f"Node '{node_name}' already exists")
            
            # Create node
            node = ComputeNode(
                name=node_name,
                model_path=model_path,
                graph_name=graph_name,
                inputs=inputs,
                outputs=outputs,
                dependencies=dependencies or set(),
                callback=callback,
            )
            
            # Load model into appropriate bus
            if dedicated_bus:
                node.bus = TensorBus(
                    name=f"{self.name}_{node_name}",
                    resolver=self.resolver,
                    transports=[]
                )
                node.bus.load_model(model_path)
            else:
                # Use shared blackboard
                self.blackboard.load_model(model_path)
            
            self._nodes[node_name] = node
            self._invalidate_execution_order()
            
            logger.info(
                f"Added model '{node_name}' (graph: {graph_name}) with "
                f"{len(inputs)} inputs, {len(outputs)} outputs, "
                f"{len(node.dependencies)} dependencies"
            )
            return node
    
    def add_node(
        self,
        name: str,
        model_path: Optional[str] = None,
        inputs: Optional[List[str]] = None,
        outputs: Optional[List[str]] = None,
        dependencies: Optional[Set[str]] = None,
        callback: Optional[Callable[[ComputeNode], None]] = None,
        dedicated_bus: bool = False,
    ) -> ComputeNode:
        """Add a compute node to the fabric (legacy method).
        
        Deprecated: Use add_model() instead for automatic input/output extraction.
        
        Args:
            name: Unique node name
            model_path: Path to ONNX model
            inputs: Input tensor names (deprecated, auto-extracted if None)
            outputs: Output tensor names (deprecated, auto-extracted if None)
            dependencies: Set of node names that must complete first
            callback: Optional completion callback
            dedicated_bus: If True, create dedicated TensorBus for this node
        
        Returns:
            Created ComputeNode
        """
        if model_path:
            # Use new add_model if we have a model_path
            return self.add_model(
                model_path=model_path,
                name=name,
                dependencies=dependencies,
                callback=callback,
                dedicated_bus=dedicated_bus
            )
        else:
            # Manual node creation (for custom operations)
            with self._lock:
                if name in self._nodes:
                    raise ValueError(f"Node '{name}' already exists")
                
                node = ComputeNode(
                    name=name,
                    model_path=model_path,
                    inputs=inputs or [],
                    outputs=outputs or [],
                    dependencies=dependencies or set(),
                    callback=callback,
                )
                
                self._nodes[name] = node
                self._invalidate_execution_order()
                
                logger.info(f"Added node '{name}' with {len(node.dependencies)} dependencies")
                return node
    
    def _invalidate_execution_order(self) -> None:
        """Mark execution order as needing recomputation."""
        self._execution_order = []
    
    def _compute_execution_order(self) -> List[str]:
        """Compute topological sort of compute graph.
        
        Returns:
            List of node names in execution order
            
        Raises:
            ValueError: If graph contains cycles
        """
        if self._execution_order:
            return self._execution_order
        
        # Kahn's algorithm for topological sort
        in_degree = {name: len(node.dependencies) for name, node in self._nodes.items()}
        queue = [name for name, degree in in_degree.items() if degree == 0]
        order = []
        
        while queue:
            # Sort queue for deterministic ordering
            queue.sort()
            node_name = queue.pop(0)
            order.append(node_name)
            
            # Reduce in-degree for dependent nodes
            for name, node in self._nodes.items():
                if node_name in node.dependencies:
                    in_degree[name] -= 1
                    if in_degree[name] == 0:
                        queue.append(name)
        
        # Check for cycles
        if len(order) != len(self._nodes):
            remaining = set(self._nodes.keys()) - set(order)
            raise ValueError(f"Circular dependency detected in nodes: {remaining}")
        
        self._execution_order = order
        logger.info(f"Execution order: {' → '.join(order)}")
        return order
    
    def set_tensor(self, name: str, arr: np.ndarray) -> None:
        """Set tensor value on blackboard (same contract as TensorBus).
        
        Args:
            name: Registered tensor name from model inputs/outputs
            arr: NumPy array to store
            
        Raises:
            KeyError: If tensor name not registered
            TypeError: If arr is not a numpy array
            ValueError: If array contains NaN or Inf
        """
        self.blackboard.set_tensor(name, arr)
    
    def get_tensor(self, name: str) -> TensorBinding:
        """Retrieve tensor binding by name (same contract as TensorBus).
        
        Args:
            name: Tensor name
            
        Returns:
            TensorBinding with values and metadata
            
        Raises:
            KeyError: If tensor name not found
        """
        return self.blackboard.get_tensor(name)
    
    # Convenience aliases for clearer intent
    def set_input(self, name: str, arr: np.ndarray) -> None:
        """Set input tensor on blackboard (alias for set_tensor).
        
        Args:
            name: Tensor name
            arr: NumPy array to store
        """
        self.set_tensor(name, arr)
    
    def get_output(self, name: str) -> np.ndarray:
        """Get output tensor from blackboard as NumPy array (convenience method).
        
        Args:
            name: Tensor name
            
        Returns:
            NumPy array
            
        Raises:
            KeyError: If tensor name not found or no value available
        """
        binding = self.get_tensor(name)
        value = binding.values.get("cpu")
        if value:
            return value.numpy()
        raise KeyError(f"No value for tensor '{name}'")
    
    def execute(self, timeout: Optional[float] = None) -> Dict[str, NodeState]:
        """Execute the computation graph.
        
        Args:
            timeout: Maximum execution time in seconds
            
        Returns:
            Dictionary mapping node names to final states
        """
        with self._lock:
            # Reset state
            self._completed_nodes.clear()
            self._failed_nodes.clear()
            self._active_workers.clear()
            
            for node in self._nodes.values():
                node.state = NodeState.IDLE
                node.error = None
            
            # Compute execution order
            try:
                order = self._compute_execution_order()
            except ValueError as e:
                logger.error(f"Cannot execute: {e}")
                raise
        
        # Execute nodes
        import time
        start_time = time.time()
        
        while True:
            with self._lock:
                # Check completion
                if len(self._completed_nodes) + len(self._failed_nodes) == len(self._nodes):
                    break
                
                # Check timeout
                if timeout and (time.time() - start_time) > timeout:
                    logger.error("Execution timeout")
                    break
                
                # Find nodes ready to execute
                ready_nodes = []
                for name in order:
                    node = self._nodes[name]
                    if node.state in (NodeState.COMPLETED, NodeState.RUNNING, NodeState.FAILED):
                        continue
                    
                    # Check if dependencies are met
                    deps_met = all(
                        self._nodes[dep].state == NodeState.COMPLETED
                        for dep in node.dependencies
                    )
                    
                    if deps_met:
                        # Check if inputs are available
                        inputs_ready = self._check_inputs_ready(node)
                        if inputs_ready:
                            ready_nodes.append(node)
                
                # Execute ready nodes (up to max_parallel)
                for node in ready_nodes:
                    if len(self._active_workers) >= self._max_parallel:
                        break
                    
                    if node.name not in self._active_workers:
                        self._execute_node(node)
            
            time.sleep(0.01)  # Brief sleep to avoid busy waiting
        
        # Wait for worker threads
        for thread in self._worker_threads:
            thread.join(timeout=1.0)
        self._worker_threads.clear()
        
        # Return final states
        return {name: node.state for name, node in self._nodes.items()}
    
    def _check_inputs_ready(self, node: ComputeNode) -> bool:
        """Check if all input tensors are available for a node."""
        for input_name in node.inputs:
            try:
                self.blackboard.get_tensor(input_name)
            except KeyError:
                return False
        return True
    
    def _execute_node(self, node: ComputeNode) -> None:
        """Execute a single compute node in a worker thread."""
        node.state = NodeState.RUNNING
        self._active_workers.add(node.name)
        
        thread = threading.Thread(
            target=self._run_node,
            args=(node,),
            daemon=True,
            name=f"node-{node.name}"
        )
        thread.start()
        self._worker_threads.append(thread)
    
    def _run_node(self, node: ComputeNode) -> None:
        """Worker function to run a node."""
        try:
            logger.info(f"Executing node '{node.name}'")
            
            if node.graph_name:
                # Run ONNX model
                if node.bus:
                    # Use dedicated bus
                    # Copy inputs from blackboard
                    for input_name in node.inputs:
                        binding = self.blackboard.get_tensor(input_name)
                        value = binding.values.get("cpu")
                        if value:
                            node.bus.set_tensor(input_name, value.numpy())
                    
                    # Run model using graph name
                    node.bus.run(node.graph_name)
                    
                    # Copy outputs to blackboard
                    for output_name in node.outputs:
                        binding = node.bus.get_tensor(output_name)
                        value = binding.values.get("cpu")
                        if value:
                            self.blackboard.set_tensor(output_name, value.numpy())
                else:
                    # Use shared blackboard (graph name)
                    self.blackboard.run(node.graph_name)
            
            # Mark as completed
            with self._lock:
                node.state = NodeState.COMPLETED
                self._completed_nodes.add(node.name)
                self._active_workers.discard(node.name)
            
            logger.info(f"Node '{node.name}' completed successfully")
            
            # Call completion callback
            if node.callback:
                try:
                    node.callback(node)
                except Exception as e:
                    logger.error(f"Node '{node.name}' callback failed: {e}")
        
        except Exception as e:
            logger.error(f"Node '{node.name}' failed: {e}")
            
            with self._lock:
                node.retries += 1
                
                # Retry logic
                if node.retries < node.max_retries:
                    logger.info(f"Retrying node '{node.name}' ({node.retries}/{node.max_retries})")
                    node.state = NodeState.IDLE
                    self._active_workers.discard(node.name)
                else:
                    node.state = NodeState.FAILED
                    node.error = str(e)
                    self._failed_nodes.add(node.name)
                    self._active_workers.discard(node.name)
    
    def get_node(self, name: str) -> ComputeNode:
        """Get a compute node by name."""
        with self._lock:
            if name not in self._nodes:
                raise KeyError(f"Node '{name}' not found")
            return self._nodes[name]
    
    def get_graph_status(self) -> Dict[str, Any]:
        """Get current status of the computation graph."""
        with self._lock:
            return {
                "total_nodes": len(self._nodes),
                "completed": len(self._completed_nodes),
                "failed": len(self._failed_nodes),
                "active": len(self._active_workers),
                "nodes": {
                    name: {
                        "state": node.state.value,
                        "retries": node.retries,
                        "error": node.error,
                    }
                    for name, node in self._nodes.items()
                }
            }
