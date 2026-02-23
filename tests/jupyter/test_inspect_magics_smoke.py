import pytest
np = pytest.importorskip("numpy")
IPY = pytest.importorskip("IPython")
from IPython import InteractiveShell  # noqa: E402

from src.jupyter.inspect.magics import load_ipython_extension
from src.jupyter.inspect.decoders import ImageView, EmbeddingView, AttentionView


def _clear_ns(ip):
    for k in list(ip.user_ns.keys()):
        if k.startswith("_fuse_") or k in ("img", "emb", "att"):
            ip.user_ns.pop(k, None)


def test_image_magic_and_inspect_auto_detect():
    ip = InteractiveShell.instance()
    load_ipython_extension(ip)
    _clear_ns(ip)

    img = (np.random.rand(8, 8, 3) * 255).astype("uint8")
    ip.user_ns["img"] = img

    # Call %image
    res = ip.run_line_magic("image", "img")
    assert isinstance(res, ImageView)

    # %inspect should auto-detect image (or explicit as image)
    res2 = ip.run_line_magic("inspect", "img as image")
    assert isinstance(res2, ImageView)


def test_pca_magic_embeddings():
    ip = InteractiveShell.instance()
    load_ipython_extension(ip)
    _clear_ns(ip)

    emb = np.random.randn(120, 32).astype("float32")
    ip.user_ns["emb"] = emb

    res = ip.run_line_magic("pca", "emb")
    assert isinstance(res, EmbeddingView)


def test_attention_magic():
    ip = InteractiveShell.instance()
    load_ipython_extension(ip)
    _clear_ns(ip)

    # heads x seq x seq
    att = np.random.rand(4, 12, 12).astype("float32")
    ip.user_ns["att"] = att

    res = ip.run_line_magic("attention", "att")
    assert isinstance(res, AttentionView)
