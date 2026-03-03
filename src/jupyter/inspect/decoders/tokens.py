from __future__ import annotations
import html as _html
import json
from typing import Dict, List, Optional, Union


from ..core import TensorView, TensorLike
from ..registry import register_decoder

@register_decoder("tokens")
def as_tokens(
    tensor: TensorLike,
    name: Optional[str] = None,
    vocab: Optional[Union[str, Dict[int, str], List[str]]] = None,
    tokenizer: Optional[str] = None,
    max_tokens: int = 100,
    show_ids: bool = True,
) -> "TokenView":
    """Display tensor as decoded tokens.
    
    Args:
        tensor: Token ID tensor
        name: Display name
        vocab: Vocabulary (dict mapping id->str, list, or path to JSON)
        tokenizer: Tokenizer name (e.g., 'gpt2', 'bert-base-uncased')
        max_tokens: Maximum tokens to display
        show_ids: Show token IDs below tokens
    """
    return TokenView(
        tensor, name, vocab=vocab, tokenizer=tokenizer,
        max_tokens=max_tokens, show_ids=show_ids
    )


class TokenView(TensorView):
    """Token tensor visualization with decoded text."""
    
    # Simple built-in vocabularies for common special tokens
    COMMON_TOKENS = {
        0: "[PAD]", 1: "[UNK]", 2: "[CLS]", 3: "[SEP]", 4: "[MASK]",
        101: "[CLS]", 102: "[SEP]", 103: "[MASK]",
    }
    
    def __init__(self, tensor: TensorLike, name: Optional[str] = None, **kwargs):
        super().__init__(tensor, name or "tokens")
        self.vocab = kwargs.get("vocab")
        self.tokenizer_name = kwargs.get("tokenizer")
        self.max_tokens = kwargs.get("max_tokens", 100)
        self.show_ids = kwargs.get("show_ids", True)
        self._tokenizer = None
        self._vocab_dict: Optional[Dict[int, str]] = None
    
    def _load_vocab(self) -> Dict[int, str]:
        """Load vocabulary for decoding."""
        if self._vocab_dict is not None:
            return self._vocab_dict
        
        # Try to load tokenizer
        if self.tokenizer_name:
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
                self._vocab_dict = {v: k for k, v in self._tokenizer.get_vocab().items()}
                return self._vocab_dict
            except Exception:
                pass
        
        # Use provided vocab
        if isinstance(self.vocab, dict):
            self._vocab_dict = self.vocab
        elif isinstance(self.vocab, list):
            self._vocab_dict = {i: t for i, t in enumerate(self.vocab)}
        elif isinstance(self.vocab, str):
            try:
                with open(self.vocab) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self._vocab_dict = {i: t for i, t in enumerate(data)}
                else:
                    self._vocab_dict = {int(k): v for k, v in data.items()}
            except Exception:
                self._vocab_dict = {}
        else:
            self._vocab_dict = {}
        
        return self._vocab_dict
    
    def _decode_token(self, token_id: int) -> str:
        """Decode a single token ID to string."""
        vocab = self._load_vocab()
        
        if self._tokenizer:
            try:
                return self._tokenizer.decode([token_id])
            except Exception:
                pass
        
        if token_id in vocab:
            return vocab[token_id]
        if token_id in self.COMMON_TOKENS:
            return self.COMMON_TOKENS[token_id]
        
        return f"[{token_id}]"
    
    def _get_token_type(self, token_id: int, text: str) -> str:
        """Classify token type for coloring."""
        if text.startswith("[") and text.endswith("]"):
            return "special"
        if text.startswith("##") or text.startswith("▁"):
            return "subword"
        return "word"
    
    def decode(self) -> str:
        """Decode all tokens to string."""
        arr = self._array.flatten().astype(int)
        tokens = [self._decode_token(int(t)) for t in arr]
        return " ".join(tokens).replace(" ##", "").replace("▁", " ")
    
    def _repr_html_(self) -> str:
        arr = self._array.flatten().astype(int)
        n_tokens = min(len(arr), self.max_tokens)
        
        token_html_parts = []
        type_colors = {
            "word": "#667eea",
            "subword": "#e67e22",
            "special": "#e74c3c",
        }
        
        for i in range(n_tokens):
            token_id = int(arr[i])
            text = self._decode_token(token_id)
            token_type = self._get_token_type(token_id, text)
            color = type_colors.get(token_type, "#667eea")
            
            id_part = f'<div style="font-size: 9px; color: #999;">{token_id}</div>' if self.show_ids else ""
            
            token_html_parts.append(f"""
            <div style="display: inline-flex; flex-direction: column; align-items: center;
                        margin: 2px; padding: 2px 6px; background: {color}22; border-radius: 4px;
                        border: 1px solid {color}44;">
                <div style="font-size: 12px; font-weight: 500; color: {color};">{_html.escape(text)}</div>
                {id_part}
            </div>
            """)
        
        if len(arr) > self.max_tokens:
            token_html_parts.append(f'<div style="padding: 4px; color: #999;">... +{len(arr) - self.max_tokens} more</div>')
        
        vocab_info = self.tokenizer_name or ("custom vocab" if self.vocab else "no vocab")
        
        return f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    border: 1px solid #e1e4e8; border-radius: 8px; padding: 12px; margin: 8px 0;
                    background: linear-gradient(135deg, #fef9f3 0%, #fff 100%);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-weight: 600; font-size: 14px;">📝 {_html.escape(self.name)}</span>
                <span style="font-size: 11px; color: #666;">
                    {len(arr)} tokens | {vocab_info}
                </span>
            </div>
            <div style="display: flex; flex-wrap: wrap; gap: 2px; margin-bottom: 8px;">
                {"".join(token_html_parts)}
            </div>
            <div style="font-size: 11px; color: #999;">
                Legend: <span style="color: #667eea;">■ word</span> 
                <span style="color: #e67e22;">■ subword</span>
                <span style="color: #e74c3c;">■ special</span>
            </div>
        </div>
        """

