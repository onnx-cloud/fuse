"""IPython magics for tensor inspection.

Provides user-friendly magic commands:
- %inspect x - Universal tensor inspection
- %inspect x as image - Force specific decoder
- %graph - Display current model graph
- %stats x - Show tensor statistics
- %compare a b - Compare two tensors
"""

from __future__ import annotations

from typing import Any

try:
    from IPython.core.magic import Magics, magics_class, line_magic, cell_magic
    from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring
    from IPython import get_ipython
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False
    # Stub for non-IPython environments
    def magics_class(cls):
        return cls
    class Magics:
        pass
    def line_magic(f):
        return f
    def cell_magic(f):
        return f
    def magic_arguments(f):
        return f
    def argument(*args, **kwargs):
        def decorator(f):
            return f
        return decorator
    def parse_argstring(f, s):
        return s


@magics_class
class InspectMagics(Magics):
    """IPython magics for tensor inspection."""
    
    @line_magic
    @magic_arguments()
    @argument('expr', nargs='*', help='Expression to inspect')
    @argument('--as', '-a', dest='decoder', help='Decoder to use (image, audio, tokens, etc.)')
    @argument('--name', '-n', help='Display name')
    def inspect(self, line: str) -> Any:
        """Inspect a tensor with auto-detection or specified decoder.
        
        Usage:
            %inspect x          # Auto-detect best visualization
            %inspect x as image # Force image decoder
            %inspect x -a audio # Same, using flag
            %inspect x --name "My Tensor"
        """
        from .core import inspect as inspect_fn
        from .registry import get_decoder
        
        # Parse "x as decoder" syntax
        parts = line.split()
        expr = None
        decoder = None
        name = None
        
        i = 0
        while i < len(parts):
            if parts[i] == 'as' and i + 1 < len(parts):
                decoder = parts[i + 1]
                i += 2
            elif parts[i] in ('--as', '-a') and i + 1 < len(parts):
                decoder = parts[i + 1]
                i += 2
            elif parts[i] in ('--name', '-n') and i + 1 < len(parts):
                name = parts[i + 1]
                i += 2
            elif expr is None:
                expr = parts[i]
                i += 1
            else:
                i += 1
        
        if not expr:
            print("Usage: %inspect <expr> [as <decoder>] [--name <name>]")
            return
        
        # Evaluate expression in user namespace
        try:
            tensor = self.shell.ev(expr)
        except Exception as e:
            print(f"Error evaluating '{expr}': {e}")
            return
        
        # Apply decoder or auto-inspect
        if decoder:
            decoder_fn = get_decoder(decoder)
            if decoder_fn:
                return decoder_fn(tensor, name=name or expr)
            else:
                print(f"Unknown decoder: {decoder}")
                print("Available: image, audio, tokens, attention, embeddings, video, points, boxes")
                return
        else:
            return inspect_fn(tensor, name=name or expr)
    
    @line_magic
    def stats(self, line: str) -> Any:
        """Show comprehensive statistics for a tensor.
        
        Usage:
            %stats x
        """
        from .analysis import describe
        
        expr = line.strip()
        if not expr:
            print("Usage: %stats <expr>")
            return
        
        try:
            tensor = self.shell.ev(expr)
        except Exception as e:
            print(f"Error evaluating '{expr}': {e}")
            return
        
        return describe(tensor, name=expr)
    
    @line_magic
    def histogram(self, line: str) -> Any:
        """Show histogram of tensor values.
        
        Usage:
            %histogram x
            %histogram x 100  # 100 bins
        """
        from .analysis import histogram
        
        parts = line.strip().split()
        if not parts:
            print("Usage: %histogram <expr> [bins]")
            return
        
        expr = parts[0]
        bins = int(parts[1]) if len(parts) > 1 else 50
        
        try:
            tensor = self.shell.ev(expr)
        except Exception as e:
            print(f"Error evaluating '{expr}': {e}")
            return
        
        return histogram(tensor, name=expr, bins=bins)
    
    @line_magic
    @magic_arguments()
    @argument('a', help='First tensor expression')
    @argument('b', nargs='?', help='Second tensor expression')
    @argument('--atol', type=float, default=1e-5, help='Absolute tolerance')
    @argument('--rtol', type=float, default=1e-5, help='Relative tolerance')
    def compare(self, line: str) -> Any:
        """Compare two tensors.
        
        Usage:
            %compare a b
            %compare a b --atol 1e-4
        """
        from .analysis import compare_tensors
        
        parts = line.strip().split()
        if len(parts) < 2:
            print("Usage: %compare <expr_a> <expr_b> [--atol N] [--rtol N]")
            return
        
        expr_a = parts[0]
        expr_b = parts[1]
        
        # Parse optional args
        atol = 1e-5
        rtol = 1e-5
        i = 2
        while i < len(parts):
            if parts[i] == '--atol' and i + 1 < len(parts):
                atol = float(parts[i + 1])
                i += 2
            elif parts[i] == '--rtol' and i + 1 < len(parts):
                rtol = float(parts[i + 1])
                i += 2
            else:
                i += 1
        
        try:
            tensor_a = self.shell.ev(expr_a)
            tensor_b = self.shell.ev(expr_b)
        except Exception as e:
            print(f"Error evaluating expressions: {e}")
            return
        
        return compare_tensors(tensor_a, tensor_b, name_a=expr_a, name_b=expr_b, atol=atol, rtol=rtol)
    
    @line_magic
    def pca(self, line: str) -> Any:
        """Project tensor using PCA and visualize.
        
        Usage:
            %pca embeddings
            %pca embeddings 3  # 3D
        """
        from .decoders import as_embeddings
        
        parts = line.strip().split()
        if not parts:
            print("Usage: %pca <expr> [n_components]")
            return
        
        expr = parts[0]
        n_components = int(parts[1]) if len(parts) > 1 else 2
        
        try:
            tensor = self.shell.ev(expr)
        except Exception as e:
            print(f"Error evaluating '{expr}': {e}")
            return
        
        return as_embeddings(tensor, name=expr, method="pca", n_components=n_components)
    
    @line_magic
    def tsne(self, line: str) -> Any:
        """Project tensor using t-SNE and visualize.
        
        Usage:
            %tsne embeddings
            %tsne embeddings 30  # perplexity=30
        """
        from .decoders import as_embeddings
        
        parts = line.strip().split()
        if not parts:
            print("Usage: %tsne <expr> [perplexity]")
            return
        
        expr = parts[0]
        perplexity = int(parts[1]) if len(parts) > 1 else 30
        
        try:
            tensor = self.shell.ev(expr)
        except Exception as e:
            print(f"Error evaluating '{expr}': {e}")
            return
        
        return as_embeddings(tensor, name=expr, method="tsne", perplexity=perplexity)
    
    @line_magic
    def image(self, line: str) -> Any:
        """Display tensor as image(s).
        
        Usage:
            %image x
            %image x --imagenet  # ImageNet denormalization
        """
        from .decoders import as_image
        
        parts = line.strip().split()
        if not parts:
            print("Usage: %image <expr> [--imagenet]")
            return
        
        expr = parts[0]
        imagenet = '--imagenet' in parts
        
        try:
            tensor = self.shell.ev(expr)
        except Exception as e:
            print(f"Error evaluating '{expr}': {e}")
            return
        
        return as_image(tensor, name=expr, imagenet=imagenet)
    
    @line_magic
    def audio(self, line: str) -> Any:
        """Display tensor as audio.
        
        Usage:
            %audio x
            %audio x 22050  # sample rate
        """
        from .decoders import as_audio
        
        parts = line.strip().split()
        if not parts:
            print("Usage: %audio <expr> [sample_rate]")
            return
        
        expr = parts[0]
        sample_rate = int(parts[1]) if len(parts) > 1 else 16000
        
        try:
            tensor = self.shell.ev(expr)
        except Exception as e:
            print(f"Error evaluating '{expr}': {e}")
            return
        
        return as_audio(tensor, name=expr, sample_rate=sample_rate)
    
    @line_magic
    def attention(self, line: str) -> Any:
        """Display attention matrix.
        
        Usage:
            %attention attn_weights
            %attention attn_weights --head 0
        """
        from .decoders import as_attention
        
        parts = line.strip().split()
        if not parts:
            print("Usage: %attention <expr> [--head N]")
            return
        
        expr = parts[0]
        head = None
        
        i = 1
        while i < len(parts):
            if parts[i] == '--head' and i + 1 < len(parts):
                head = int(parts[i + 1])
                i += 2
            else:
                i += 1
        
        try:
            tensor = self.shell.ev(expr)
        except Exception as e:
            print(f"Error evaluating '{expr}': {e}")
            return
        
        return as_attention(tensor, name=expr, head=head)
    
    @line_magic 
    def tokens(self, line: str) -> Any:
        """Display tensor as decoded tokens.
        
        Usage:
            %tokens x
            %tokens x --tokenizer gpt2
        """
        from .decoders import as_tokens
        
        parts = line.strip().split()
        if not parts:
            print("Usage: %tokens <expr> [--tokenizer NAME]")
            return
        
        expr = parts[0]
        tokenizer = None
        
        i = 1
        while i < len(parts):
            if parts[i] == '--tokenizer' and i + 1 < len(parts):
                tokenizer = parts[i + 1]
                i += 2
            else:
                i += 1
        
        try:
            tensor = self.shell.ev(expr)
        except Exception as e:
            print(f"Error evaluating '{expr}': {e}")
            return
        
        return as_tokens(tensor, name=expr, tokenizer=tokenizer)
    
    # =========================================================================
    # Graph & Model Inspection Magics
    # =========================================================================
    
    @line_magic
    def graph(self, line: str) -> Any:
        """Display model graph visualization.
        
        Usage:
            %graph model
            %graph model --max-nodes 50
        """
        from .graph import graph as graph_fn
        
        parts = line.strip().split()
        if not parts:
            print("Usage: %graph <model> [--max-nodes N]")
            return
        
        expr = parts[0]
        max_nodes = 100
        
        i = 1
        while i < len(parts):
            if parts[i] == '--max-nodes' and i + 1 < len(parts):
                max_nodes = int(parts[i + 1])
                i += 2
            else:
                i += 1
        
        try:
            model = self.shell.ev(expr)
        except Exception as e:
            print(f"Error evaluating '{expr}': {e}")
            return
        
        return graph_fn(model, name=expr, max_nodes=max_nodes)
    
    @line_magic
    def diff(self, line: str) -> Any:
        """Compare two model graphs.
        
        Usage:
            %diff model1 model2
        """
        from .graph import graph_diff
        
        parts = line.strip().split()
        if len(parts) < 2:
            print("Usage: %diff <model1> <model2>")
            return
        
        expr_a, expr_b = parts[0], parts[1]
        
        try:
            model_a = self.shell.ev(expr_a)
            model_b = self.shell.ev(expr_b)
        except Exception as e:
            print(f"Error evaluating models: {e}")
            return
        
        return graph_diff(model_a, model_b, name_a=expr_a, name_b=expr_b)
    
    @line_magic
    def trace(self, line: str) -> Any:
        """Profile model execution.
        
        Usage:
            %trace model
            %trace model --runs 10
        """
        from .trace import trace as trace_fn
        
        parts = line.strip().split()
        if not parts:
            print("Usage: %trace <model> [--runs N]")
            return
        
        expr = parts[0]
        runs = 5
        
        i = 1
        while i < len(parts):
            if parts[i] == '--runs' and i + 1 < len(parts):
                runs = int(parts[i + 1])
                i += 2
            else:
                i += 1
        
        try:
            model = self.shell.ev(expr)
        except Exception as e:
            print(f"Error evaluating '{expr}': {e}")
            return
        
        return trace_fn(model, name=expr, n_runs=runs)
    
    @line_magic
    def weights(self, line: str) -> Any:
        """Display model weights summary.
        
        Usage:
            %weights model
            %weights model --top 20
        """
        from .weights import weights as weights_fn
        
        parts = line.strip().split()
        if not parts:
            print("Usage: %weights <model> [--top N]")
            return
        
        expr = parts[0]
        top_k = 50
        
        i = 1
        while i < len(parts):
            if parts[i] == '--top' and i + 1 < len(parts):
                top_k = int(parts[i + 1])
                i += 2
            else:
                i += 1
        
        try:
            model = self.shell.ev(expr)
        except Exception as e:
            print(f"Error evaluating '{expr}': {e}")
            return
        
        return weights_fn(model, name=expr, top_k=top_k)
    
    @line_magic
    def filters(self, line: str) -> Any:
        """Display conv filter visualization.
        
        Usage:
            %filters weight_tensor
            %filters weight_tensor --max 16
        """
        from .weights import filters as filters_fn
        
        parts = line.strip().split()
        if not parts:
            print("Usage: %filters <tensor> [--max N]")
            return
        
        expr = parts[0]
        max_filters = 16
        
        i = 1
        while i < len(parts):
            if parts[i] == '--max' and i + 1 < len(parts):
                max_filters = int(parts[i + 1])
                i += 2
            else:
                i += 1
        
        try:
            tensor = self.shell.ev(expr)
        except Exception as e:
            print(f"Error evaluating '{expr}': {e}")
            return
        
        return filters_fn(tensor, name=expr, max_filters=max_filters)
    
    @line_magic
    def slice(self, line: str) -> Any:
        """Display a slice of a tensor.
        
        Usage:
            %slice x[0, :10, :10]
            %slice x --head 5
            %slice x --tail 5
        """
        from .slice import slice_tensor, head, tail
        
        parts = line.strip().split()
        if not parts:
            print("Usage: %slice <expr> [--head N] [--tail N]")
            return
        
        expr = parts[0]
        head_n = None
        tail_n = None
        
        i = 1
        while i < len(parts):
            if parts[i] == '--head' and i + 1 < len(parts):
                head_n = int(parts[i + 1])
                i += 2
            elif parts[i] == '--tail' and i + 1 < len(parts):
                tail_n = int(parts[i + 1])
                i += 2
            else:
                i += 1
        
        try:
            tensor = self.shell.ev(expr)
        except Exception as e:
            print(f"Error evaluating '{expr}': {e}")
            return
        
        if head_n is not None:
            return head(tensor, head_n, name=expr)
        elif tail_n is not None:
            return tail(tensor, tail_n, name=expr)
        else:
            return slice_tensor(tensor, name=expr)
    
    @line_magic
    def report(self, line: str) -> Any:
        """Generate comprehensive model report.
        
        Usage:
            %report model
            %report model --save report.html
        """
        from .report import report as report_fn, export_report
        
        parts = line.strip().split()
        if not parts:
            print("Usage: %report <model> [--save path.html]")
            return
        
        expr = parts[0]
        save_path = None
        
        i = 1
        while i < len(parts):
            if parts[i] == '--save' and i + 1 < len(parts):
                save_path = parts[i + 1]
                i += 2
            else:
                i += 1
        
        try:
            model = self.shell.ev(expr)
        except Exception as e:
            print(f"Error evaluating '{expr}': {e}")
            return
        
        if save_path:
            export_report(model, save_path, name=expr)
            print(f"Report saved to {save_path}")
            return
        
        return report_fn(model, name=expr)


def load_ipython_extension(ipython):
    """Load the extension in IPython."""
    ipython.register_magics(InspectMagics)


def unload_ipython_extension(ipython):
    """Unload the extension."""
    pass


def register_magics():
    """Register magics with the current IPython instance."""
    if HAS_IPYTHON:
        ip = get_ipython()
        if ip is not None:
            ip.register_magics(InspectMagics)
            return True
    return False
