"""Statistical analysis tools for tensors.

Provides dimensionality reduction, distribution analysis,
and tensor comparison utilities.
"""

from __future__ import annotations

import html as _html
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .core import TensorView, TensorLike, _to_numpy, _format_shape, _format_dtype, _format_size

try:
    from IPython.display import HTML, display
except ImportError:
    HTML = None
    display = print


# ============================================================================
# Dimensionality Reduction
# ============================================================================

def pca(
    tensor: TensorLike,
    n_components: int = 2,
    whiten: bool = False,
) -> np.ndarray:
    """Project tensor to lower dimensions using PCA.
    
    Args:
        tensor: Input tensor (n_samples, n_features)
        n_components: Target dimensions (2 or 3)
        whiten: Whether to whiten the output
        
    Returns:
        Projected tensor (n_samples, n_components)
    """
    arr = _to_numpy(tensor)
    if arr.ndim != 2:
        arr = arr.reshape(-1, arr.shape[-1])
    
    try:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=n_components, whiten=whiten)
        return reducer.fit_transform(arr)
    except ImportError:
        # Manual PCA implementation
        arr = arr - arr.mean(axis=0)
        cov = np.cov(arr.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        projection = eigenvectors[:, :n_components]
        result = arr @ projection
        if whiten:
            result = result / np.sqrt(eigenvalues[idx[:n_components]] + 1e-8)
        return result


def tsne(
    tensor: TensorLike,
    n_components: int = 2,
    perplexity: int = 30,
    n_iter: int = 1000,
    learning_rate: Union[str, float] = "auto",
) -> np.ndarray:
    """Project tensor using t-SNE.
    
    Args:
        tensor: Input tensor (n_samples, n_features)
        n_components: Target dimensions (2 or 3)
        perplexity: t-SNE perplexity parameter
        n_iter: Number of iterations
        learning_rate: Learning rate ('auto' or float)
        
    Returns:
        Projected tensor (n_samples, n_components)
    """
    arr = _to_numpy(tensor)
    if arr.ndim != 2:
        arr = arr.reshape(-1, arr.shape[-1])
    
    try:
        from sklearn.manifold import TSNE
        reducer = TSNE(
            n_components=n_components,
            perplexity=min(perplexity, arr.shape[0] - 1),
            n_iter=n_iter,
            learning_rate=learning_rate,
            random_state=42,
        )
        return reducer.fit_transform(arr)
    except ImportError:
        # Fallback to PCA
        print("Warning: sklearn not available, falling back to PCA")
        return pca(arr, n_components=n_components)


def umap_project(
    tensor: TensorLike,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
) -> np.ndarray:
    """Project tensor using UMAP.
    
    Args:
        tensor: Input tensor (n_samples, n_features)
        n_components: Target dimensions (2 or 3)
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        metric: Distance metric
        
    Returns:
        Projected tensor (n_samples, n_components)
    """
    arr = _to_numpy(tensor)
    if arr.ndim != 2:
        arr = arr.reshape(-1, arr.shape[-1])
    
    try:
        import umap
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=min(n_neighbors, arr.shape[0] - 1),
            min_dist=min_dist,
            metric=metric,
            random_state=42,
        )
        return reducer.fit_transform(arr)
    except ImportError:
        # Fallback to t-SNE then PCA
        print("Warning: umap-learn not available, falling back to t-SNE")
        return tsne(arr, n_components=n_components)


# ============================================================================
# Distribution Analysis
# ============================================================================

def histogram(
    tensor: TensorLike,
    name: Optional[str] = None,
    bins: int = 50,
    range: Optional[Tuple[float, float]] = None,
    log_scale: bool = False,
) -> "HistogramView":
    """Create histogram visualization of tensor values.
    
    Args:
        tensor: Input tensor
        name: Display name
        bins: Number of bins
        range: Value range (min, max)
        log_scale: Use log scale for y-axis
    """
    return HistogramView(tensor, name, bins=bins, range=range, log_scale=log_scale)


class HistogramView(TensorView):
    """Histogram visualization of tensor value distribution."""
    
    def __init__(self, tensor: TensorLike, name: Optional[str] = None, **kwargs):
        super().__init__(tensor, name or "histogram")
        self.bins = kwargs.get("bins", 50)
        self.range = kwargs.get("range")
        self.log_scale = kwargs.get("log_scale", False)
    
    def _compute_histogram(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute histogram counts and bin edges."""
        arr = self._array.flatten()
        counts, edges = np.histogram(arr, bins=self.bins, range=self.range)
        if self.log_scale:
            counts = np.log1p(counts)
        return counts, edges
    
    def _make_svg(self, width: int = 400, height: int = 150) -> str:
        """Generate SVG histogram."""
        counts, edges = self._compute_histogram()
        
        margin = {"top": 10, "right": 10, "bottom": 30, "left": 40}
        plot_w = width - margin["left"] - margin["right"]
        plot_h = height - margin["top"] - margin["bottom"]
        
        # Normalize counts
        max_count = counts.max() or 1
        bar_width = plot_w / len(counts)
        
        bars = []
        for i, count in enumerate(counts):
            bar_height = (count / max_count) * plot_h
            x = margin["left"] + i * bar_width
            y = margin["top"] + plot_h - bar_height
            bars.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
                f'height="{bar_height:.1f}" fill="#667eea" fill-opacity="0.8"/>'
            )
        
        # Axes
        x_axis_y = margin["top"] + plot_h
        x_labels = [
            f'<text x="{margin["left"]}" y="{x_axis_y + 15}" font-size="10" fill="#666">{edges[0]:.2g}</text>',
            f'<text x="{margin["left"] + plot_w}" y="{x_axis_y + 15}" font-size="10" fill="#666" text-anchor="end">{edges[-1]:.2g}</text>',
        ]
        
        y_labels = [
            f'<text x="{margin["left"] - 5}" y="{margin["top"] + 5}" font-size="10" fill="#666" text-anchor="end">{int(max_count)}</text>',
            f'<text x="{margin["left"] - 5}" y="{x_axis_y}" font-size="10" fill="#666" text-anchor="end">0</text>',
        ]
        
        return f"""
        <svg width="{width}" height="{height}" style="background: #fafafa; border-radius: 4px;">
            <line x1="{margin['left']}" y1="{x_axis_y}" x2="{margin['left'] + plot_w}" y2="{x_axis_y}" stroke="#ccc"/>
            <line x1="{margin['left']}" y1="{margin['top']}" x2="{margin['left']}" y2="{x_axis_y}" stroke="#ccc"/>
            {"".join(bars)}
            {"".join(x_labels)}
            {"".join(y_labels)}
        </svg>
        """
    
    def _repr_html_(self) -> str:
        arr = self._array
        svg = self._make_svg()
        
        return f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    border: 1px solid #e1e4e8; border-radius: 8px; padding: 12px; margin: 8px 0;
                    background: linear-gradient(135deg, #f8f8fc 0%, #fff 100%);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-weight: 600; font-size: 14px;">📊 {_html.escape(self.name)}</span>
                <span style="font-size: 11px; color: #666;">
                    {self.bins} bins | {arr.size:,} values
                </span>
            </div>
            <div style="text-align: center;">
                {svg}
            </div>
            <div style="margin-top: 8px; display: flex; justify-content: space-between; font-size: 11px; color: #999;">
                <span>μ={arr.mean():.4g} σ={arr.std():.4g}</span>
                <span>[{arr.min():.4g}, {arr.max():.4g}]</span>
            </div>
        </div>
        """


# ============================================================================
# Tensor Comparison
# ============================================================================

def compare_tensors(
    a: TensorLike,
    b: TensorLike,
    name_a: str = "A",
    name_b: str = "B",
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> "ComparisonView":
    """Compare two tensors and visualize differences.
    
    Args:
        a: First tensor
        b: Second tensor
        name_a: Name for first tensor
        name_b: Name for second tensor
        atol: Absolute tolerance for allclose
        rtol: Relative tolerance for allclose
    """
    return ComparisonView(a, b, name_a=name_a, name_b=name_b, atol=atol, rtol=rtol)


class ComparisonView:
    """Tensor comparison visualization."""
    
    def __init__(
        self, a: TensorLike, b: TensorLike, **kwargs
    ):
        self._a = _to_numpy(a)
        self._b = _to_numpy(b)
        self.name_a = kwargs.get("name_a", "A")
        self.name_b = kwargs.get("name_b", "B")
        self.atol = kwargs.get("atol", 1e-5)
        self.rtol = kwargs.get("rtol", 1e-5)
    
    @property
    def shapes_match(self) -> bool:
        return self._a.shape == self._b.shape
    
    @property
    def allclose(self) -> bool:
        if not self.shapes_match:
            return False
        return np.allclose(self._a, self._b, atol=self.atol, rtol=self.rtol)
    
    @property
    def max_diff(self) -> float:
        if not self.shapes_match:
            return float("inf")
        return np.abs(self._a - self._b).max()
    
    @property
    def mean_diff(self) -> float:
        if not self.shapes_match:
            return float("inf")
        return np.abs(self._a - self._b).mean()
    
    @property
    def mse(self) -> float:
        if not self.shapes_match:
            return float("inf")
        return ((self._a - self._b) ** 2).mean()
    
    @property
    def cosine_similarity(self) -> float:
        if not self.shapes_match:
            return 0.0
        a_flat = self._a.flatten()
        b_flat = self._b.flatten()
        dot = np.dot(a_flat, b_flat)
        norm_a = np.linalg.norm(a_flat)
        norm_b = np.linalg.norm(b_flat)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
    
    def _make_diff_svg(self, width: int = 400, height: int = 100) -> str:
        """Generate SVG showing distribution of differences."""
        if not self.shapes_match:
            return '<em style="color: #999;">Shapes don\'t match - cannot show diff distribution</em>'
        
        diff = (self._a - self._b).flatten()
        
        # Create histogram of differences
        counts, edges = np.histogram(diff, bins=50)
        max_count = counts.max() or 1
        bar_width = width / len(counts)
        
        bars = []
        mid_x = width / 2
        for i, count in enumerate(counts):
            bar_height = (count / max_count) * (height - 20)
            x = i * bar_width
            y = height - 20 - bar_height
            
            # Color by sign (red for positive, blue for negative)
            mid_val = (edges[i] + edges[i+1]) / 2
            if mid_val > 0:
                color = f"rgba(231, 76, 60, {0.3 + 0.7 * abs(mid_val) / (abs(diff).max() + 1e-8):.2f})"
            else:
                color = f"rgba(52, 152, 219, {0.3 + 0.7 * abs(mid_val) / (abs(diff).max() + 1e-8):.2f})"
            
            bars.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
                f'height="{bar_height:.1f}" fill="{color}"/>'
            )
        
        # Zero line
        zero_x = width * (-edges[0]) / (edges[-1] - edges[0])
        
        return f"""
        <svg width="{width}" height="{height}" style="background: #fafafa; border-radius: 4px;">
            {"".join(bars)}
            <line x1="{zero_x:.1f}" y1="0" x2="{zero_x:.1f}" y2="{height - 20}" stroke="#333" stroke-dasharray="3,3"/>
            <text x="5" y="{height - 5}" font-size="10" fill="#666">{edges[0]:.2g}</text>
            <text x="{width - 5}" y="{height - 5}" font-size="10" fill="#666" text-anchor="end">{edges[-1]:.2g}</text>
        </svg>
        """
    
    def _repr_html_(self) -> str:
        status_icon = "✅" if self.allclose else "⚠️"
        status_color = "#28a745" if self.allclose else "#dc3545"
        status_text = "Match" if self.allclose else "Differ"
        
        shape_match = "✓" if self.shapes_match else "✗"
        
        diff_svg = self._make_diff_svg()
        
        return f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    border: 1px solid #e1e4e8; border-radius: 8px; padding: 12px; margin: 8px 0;
                    background: linear-gradient(135deg, #f8f8f8 0%, #fff 100%);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <span style="font-weight: 600; font-size: 14px;">
                    🔍 Compare: {_html.escape(self.name_a)} vs {_html.escape(self.name_b)}
                </span>
                <span style="font-size: 13px; font-weight: 600; color: {status_color};">
                    {status_icon} {status_text}
                </span>
            </div>
            
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin-bottom: 12px;">
                <div style="padding: 8px; background: #f8f9fa; border-radius: 4px;">
                    <div style="font-size: 11px; color: #666; margin-bottom: 4px;">{_html.escape(self.name_a)}</div>
                    <div style="font-size: 12px;">Shape: {_format_shape(self._a.shape)}</div>
                    <div style="font-size: 11px; color: #999;">μ={self._a.mean():.4g} σ={self._a.std():.4g}</div>
                </div>
                <div style="padding: 8px; background: #f8f9fa; border-radius: 4px;">
                    <div style="font-size: 11px; color: #666; margin-bottom: 4px;">{_html.escape(self.name_b)}</div>
                    <div style="font-size: 12px;">Shape: {_format_shape(self._b.shape)}</div>
                    <div style="font-size: 11px; color: #999;">μ={self._b.mean():.4g} σ={self._b.std():.4g}</div>
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-bottom: 12px;">
                <div style="text-align: center; padding: 6px; background: #fff; border: 1px solid #eee; border-radius: 4px;">
                    <div style="font-size: 10px; color: #999;">Max Diff</div>
                    <div style="font-size: 13px; font-weight: 600;">{self.max_diff:.4g}</div>
                </div>
                <div style="text-align: center; padding: 6px; background: #fff; border: 1px solid #eee; border-radius: 4px;">
                    <div style="font-size: 10px; color: #999;">Mean Diff</div>
                    <div style="font-size: 13px; font-weight: 600;">{self.mean_diff:.4g}</div>
                </div>
                <div style="text-align: center; padding: 6px; background: #fff; border: 1px solid #eee; border-radius: 4px;">
                    <div style="font-size: 10px; color: #999;">MSE</div>
                    <div style="font-size: 13px; font-weight: 600;">{self.mse:.4g}</div>
                </div>
                <div style="text-align: center; padding: 6px; background: #fff; border: 1px solid #eee; border-radius: 4px;">
                    <div style="font-size: 10px; color: #999;">Cosine Sim</div>
                    <div style="font-size: 13px; font-weight: 600;">{self.cosine_similarity:.4f}</div>
                </div>
            </div>
            
            <div style="margin-top: 8px;">
                <div style="font-size: 11px; color: #666; margin-bottom: 4px;">Difference Distribution (A - B):</div>
                {diff_svg}
            </div>
        </div>
        """


# ============================================================================
# Statistical Summaries
# ============================================================================

def describe(tensor: TensorLike, name: Optional[str] = None) -> "DescribeView":
    """Generate comprehensive statistical summary.
    
    Args:
        tensor: Input tensor
        name: Display name
    """
    return DescribeView(tensor, name)


class DescribeView(TensorView):
    """Comprehensive statistical summary view."""
    
    def __init__(self, tensor: TensorLike, name: Optional[str] = None):
        super().__init__(tensor, name or "tensor")
    
    def _compute_stats(self) -> Dict[str, Any]:
        """Compute comprehensive statistics."""
        arr = self._array.flatten()
        
        stats = {
            "count": len(arr),
            "mean": arr.mean(),
            "std": arr.std(),
            "min": arr.min(),
            "max": arr.max(),
            "median": np.median(arr),
            "q25": np.percentile(arr, 25),
            "q75": np.percentile(arr, 75),
            "zeros": (arr == 0).sum(),
            "nans": np.isnan(arr).sum(),
            "infs": np.isinf(arr).sum(),
            "unique": len(np.unique(arr)) if len(arr) < 10000 else "~",
        }
        
        # Sparsity
        stats["sparsity"] = stats["zeros"] / stats["count"] if stats["count"] > 0 else 0
        
        return stats
    
    def _repr_html_(self) -> str:
        stats = self._compute_stats()
        
        # Health indicators
        health_issues = []
        if stats["nans"] > 0:
            health_issues.append(f"⚠️ {stats['nans']} NaN values")
        if stats["infs"] > 0:
            health_issues.append(f"⚠️ {stats['infs']} Inf values")
        if stats["sparsity"] > 0.9:
            health_issues.append(f"📉 {stats['sparsity']*100:.1f}% sparse")
        
        health_html = " ".join(health_issues) if health_issues else "✅ Healthy"
        
        return f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    border: 1px solid #e1e4e8; border-radius: 8px; padding: 12px; margin: 8px 0;
                    background: linear-gradient(135deg, #f5f8ff 0%, #fff 100%);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <span style="font-weight: 600; font-size: 14px;">📋 {_html.escape(self.name)}</span>
                <span style="font-size: 11px; color: #666;">{health_html}</span>
            </div>
            
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px;">
                <div style="text-align: center; padding: 6px; background: #f8f9fa; border-radius: 4px;">
                    <div style="font-size: 10px; color: #999;">Shape</div>
                    <div style="font-size: 12px; font-weight: 500;">{_format_shape(self._array.shape)}</div>
                </div>
                <div style="text-align: center; padding: 6px; background: #f8f9fa; border-radius: 4px;">
                    <div style="font-size: 10px; color: #999;">Dtype</div>
                    <div style="font-size: 12px; font-weight: 500;">{_format_dtype(self._array.dtype)}</div>
                </div>
                <div style="text-align: center; padding: 6px; background: #f8f9fa; border-radius: 4px;">
                    <div style="font-size: 10px; color: #999;">Size</div>
                    <div style="font-size: 12px; font-weight: 500;">{_format_size(self._array.nbytes)}</div>
                </div>
                <div style="text-align: center; padding: 6px; background: #f8f9fa; border-radius: 4px;">
                    <div style="font-size: 10px; color: #999;">Elements</div>
                    <div style="font-size: 12px; font-weight: 500;">{stats['count']:,}</div>
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; margin-top: 8px;">
                <div style="text-align: center; padding: 4px;">
                    <div style="font-size: 10px; color: #999;">Min</div>
                    <div style="font-size: 11px;">{stats['min']:.4g}</div>
                </div>
                <div style="text-align: center; padding: 4px;">
                    <div style="font-size: 10px; color: #999;">Q25</div>
                    <div style="font-size: 11px;">{stats['q25']:.4g}</div>
                </div>
                <div style="text-align: center; padding: 4px;">
                    <div style="font-size: 10px; color: #999;">Median</div>
                    <div style="font-size: 11px;">{stats['median']:.4g}</div>
                </div>
                <div style="text-align: center; padding: 4px;">
                    <div style="font-size: 10px; color: #999;">Q75</div>
                    <div style="font-size: 11px;">{stats['q75']:.4g}</div>
                </div>
                <div style="text-align: center; padding: 4px;">
                    <div style="font-size: 10px; color: #999;">Max</div>
                    <div style="font-size: 11px;">{stats['max']:.4g}</div>
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 8px;">
                <div style="text-align: center; padding: 4px;">
                    <div style="font-size: 10px; color: #999;">Mean</div>
                    <div style="font-size: 11px;">{stats['mean']:.4g}</div>
                </div>
                <div style="text-align: center; padding: 4px;">
                    <div style="font-size: 10px; color: #999;">Std</div>
                    <div style="font-size: 11px;">{stats['std']:.4g}</div>
                </div>
                <div style="text-align: center; padding: 4px;">
                    <div style="font-size: 10px; color: #999;">Unique</div>
                    <div style="font-size: 11px;">{stats['unique']}</div>
                </div>
            </div>
        </div>
        """


# ============================================================================
# Outlier Detection
# ============================================================================

def detect_outliers(
    tensor: TensorLike,
    method: str = "zscore",
    threshold: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect outliers in tensor values.
    
    Args:
        tensor: Input tensor
        method: Detection method ('zscore', 'iqr', 'mad')
        threshold: Threshold for outlier detection
        
    Returns:
        Tuple of (mask of outliers, outlier values)
    """
    arr = _to_numpy(tensor).flatten()
    
    if method == "zscore":
        z = np.abs((arr - arr.mean()) / (arr.std() + 1e-8))
        mask = z > threshold
    elif method == "iqr":
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        mask = (arr < lower) | (arr > upper)
    elif method == "mad":
        median = np.median(arr)
        mad = np.median(np.abs(arr - median))
        modified_z = 0.6745 * (arr - median) / (mad + 1e-8)
        mask = np.abs(modified_z) > threshold
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return mask, arr[mask]


# ============================================================================
# Correlation Analysis
# ============================================================================

def correlation_matrix(
    tensor: TensorLike,
    feature_names: Optional[List[str]] = None,
) -> "CorrelationView":
    """Compute and visualize correlation matrix.
    
    Args:
        tensor: Input tensor (n_samples, n_features)
        feature_names: Names for each feature
    """
    return CorrelationView(tensor, feature_names=feature_names)


class CorrelationView(TensorView):
    """Correlation matrix visualization."""
    
    def __init__(self, tensor: TensorLike, **kwargs):
        super().__init__(tensor, "correlation")
        self.feature_names = kwargs.get("feature_names")
    
    def _compute_correlation(self) -> np.ndarray:
        """Compute correlation matrix."""
        arr = self._array
        if arr.ndim != 2:
            arr = arr.reshape(-1, arr.shape[-1])
        return np.corrcoef(arr.T)
    
    def _make_heatmap_svg(self, width: int = 300, height: int = 300) -> str:
        """Generate SVG correlation heatmap."""
        corr = self._compute_correlation()
        n = corr.shape[0]
        cell_size = min(width, height) / n
        
        cells = []
        for i in range(n):
            for j in range(n):
                val = corr[i, j]
                # Color scale: blue (-1) to white (0) to red (+1)
                if val >= 0:
                    r = 255
                    g = b = int(255 * (1 - val))
                else:
                    b = 255
                    r = g = int(255 * (1 + val))
                
                x, y = j * cell_size, i * cell_size
                cells.append(
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_size:.1f}" '
                    f'height="{cell_size:.1f}" fill="rgb({r},{g},{b})">'
                    f'<title>({i},{j}): {val:.3f}</title></rect>'
                )
        
        return f"""
        <svg width="{width}" height="{height}" style="border-radius: 4px;">
            {"".join(cells)}
        </svg>
        """
    
    def _repr_html_(self) -> str:
        corr = self._compute_correlation()
        n = corr.shape[0]
        svg = self._make_heatmap_svg()
        
        return f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    border: 1px solid #e1e4e8; border-radius: 8px; padding: 12px; margin: 8px 0;
                    background: linear-gradient(135deg, #fff8f8 0%, #fff 100%);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-weight: 600; font-size: 14px;">🔗 Correlation Matrix</span>
                <span style="font-size: 11px; color: #666;">{n} × {n} features</span>
            </div>
            <div style="text-align: center;">
                {svg}
            </div>
            <div style="margin-top: 8px; font-size: 11px; color: #999; text-align: center;">
                <span style="color: #3498db;">■ -1</span> ← Correlation → <span style="color: #e74c3c;">■ +1</span>
            </div>
        </div>
        """
