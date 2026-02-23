#!/usr/bin/env python3
"""
build_clip_golden.py
Creates a real ONNX file that matches the earlier ‘golden’ textual sketch
and embeds all requested metadata.
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper as nh

# ---------- config ----------
PATCH_SIZE  = 32
IMG_SIZE    = 224
N_PATCHES   = (IMG_SIZE // PATCH_SIZE) ** 2
SEQ_IMG     = N_PATCHES + 1          # 49 patches + CLS
SEQ_TXT     = 77
D_IMG       = 768
D_TXT       = 512
D_MLP       = 3072
N_LAYERS    = 12
VOCAB       = 49408
OPSSET      = 18
DOMAIN      = "examples.golden.clip"
FUSE        = "0.7"
ID_URL      = "https://ns.onnx.cloud/examples.golden.clip.full"

# ---------- helpers ----------
def rand(*sh): return np.random.randn(*sh).astype(np.float32)
def const(name, tensor): return helper.make_node("Constant", [], [name], value=nh.from_array(tensor, name=name))

# ---------- GELU ----------
def make_gelu(g, x, name):
    c1  = const(name+"_c1", np.array(0.044715, dtype=np.float32))
    c2  = const(name+"_c2", np.array(np.sqrt(2/np.pi), dtype=np.float32))
    c3  = const(name+"_c3", np.array(1.0, dtype=np.float32))
    c05 = const(name+"_c05", np.array(0.5, dtype=np.float32))
    pow3 = helper.make_node("Pow", [x, c3.output[0]], [name+"_pow3"])
    mul1 = helper.make_node("Mul", [x, c2.output[0]], [name+"_mul1"])
    mul2 = helper.make_node("Mul", [pow3.output[0], c1.output[0]], [name+"_mul2"])
    add  = helper.make_node("Add", [mul1.output[0], mul2.output[0]], [name+"_add"])
    erf  = helper.make_node("Erf", [add.output[0]], [name+"_erf"])
    add2 = helper.make_node("Add", [erf.output[0], c3.output[0]], [name+"_add2"])
    mul3 = helper.make_node("Mul", [x, add2.output[0]], [name+"_mul3"])
    return helper.make_node("Mul", [mul3.output[0], c05.output[0]], [g])

# ---------- Transformer block ----------
def transformer_block(hidden_in, prefix):
    nodes = []
    # LN1
    ln1 = helper.make_node("LayerNormalization", [hidden_in], [prefix+"_ln1"], axis=-1, epsilon=1e-5)
    nodes.append(ln1)
    # Self-attention
    for w, out in zip(["Wq","Wk","Wv"], ["q","k","v"]):
        nodes.append(helper.make_node("MatMul", [ln1.output[0], prefix+"_"+w], [prefix+"_"+out]))
    scale = const(prefix+"_scale", np.array(1/np.sqrt(D_IMG), dtype=np.float32))
    nodes.append(scale)
    scores = helper.make_node("MatMul", [prefix+"_q", prefix+"_k"], [prefix+"_scores"], transB=1)
    nodes.append(scores)
    scaled = helper.make_node("Mul", [scores.output[0], scale.output[0]], [prefix+"_scaled"])
    nodes.append(scaled)
    attn_w = helper.make_node("Softmax", [scaled.output[0]], [prefix+"_attn_w"], axis=-1)
    nodes.append(attn_w)
    attn_out = helper.make_node("MatMul", [attn_w.output[0], prefix+"_v"], [prefix+"_attn_out"])
    nodes.append(attn_out)
    proj = helper.make_node("MatMul", [attn_out.output[0], prefix+"_Wo"], [prefix+"_proj"])
    nodes.append(proj)

    add1 = helper.make_node("Add", [hidden_in, proj.output[0]], [prefix+"_add1"])
    nodes.append(add1)
    # LN2
    ln2 = helper.make_node("LayerNormalization", [add1.output[0]], [prefix+"_ln2"], axis=-1, epsilon=1e-5)
    nodes.append(ln2)
    # MLP
    mlp1 = helper.make_node("MatMul", [ln2.output[0], prefix+"_Wm1"], [prefix+"_mlp1"])
    nodes.append(mlp1)
    mlp1_b = helper.make_node("Add", [mlp1.output[0], prefix+"_bm1"], [prefix+"_mlp1_b"])
    nodes.append(mlp1_b)
    gelu_n = make_gelu(prefix+"_gelu", mlp1_b.output[0], prefix)
    nodes.append(gelu_n)
    mlp2 = helper.make_node("MatMul", [gelu_n.output[0], prefix+"_Wm2"], [prefix+"_mlp2"])
    nodes.append(mlp2)
    mlp2_b = helper.make_node("Add", [mlp2.output[0], prefix+"_bm2"], [prefix+"_mlp2_b"])
    nodes.append(mlp2_b)
    out = helper.make_node("Add", [add1.output[0], mlp2_b.output[0]], [prefix+"_out"])
    nodes.append(out)
    return nodes, out.output[0]

# ---------- Image encoder ----------
def build_image_encoder(img):
    nodes, init = [], []
    # patch projector (simplified as Conv)
    W_patch = rand(D_IMG, 3*PATCH_SIZE*PATCH_SIZE)
    b_patch = rand(D_IMG)
    init.append(nh.from_array(W_patch, name="W_patch"))
    init.append(nh.from_array(b_patch, name="b_patch"))
    proj = helper.make_node("Conv", [img, "W_patch"], ["proj"], kernel_shape=[PATCH_SIZE,PATCH_SIZE], strides=[PATCH_SIZE,PATCH_SIZE])
    proj_b = helper.make_node("Add", ["proj", "b_patch"], ["proj_b"])
    nodes.extend([proj, proj_b])
    # CLS token
    CLS = rand(1, D_IMG)
    init.append(nh.from_array(CLS, name="CLS"))
    cls_tiled = helper.make_node("Tile", ["CLS", "ones"], ["cls_tiled"])
    ones = np.array([1,1,1], dtype=np.int64)
    init.append(nh.from_array(ones, name="ones"))
    seq = helper.make_node("Concat", ["cls_tiled", "proj_b"], ["seq"], axis=1)
    nodes.extend([cls_tiled, seq])
    # positional
    P_img = rand(SEQ_IMG, D_IMG)
    init.append(nh.from_array(P_img, name="P_img"))
    seq_pos = helper.make_node("Add", ["seq", "P_img"], ["seq_pos"])
    nodes.append(seq_pos)
    # 12 layers
    h = "seq_pos"
    for lyr in range(N_LAYERS):
        blk_nodes, h = transformer_block(h, f"img_l{lyr}")
        nodes.extend(blk_nodes)
    # CLS pooling
    cls_emb = helper.make_node("Gather", [h, "zero"], ["cls_emb"], axis=1)
    zero = np.array([0], dtype=np.int64)
    init.append(nh.from_array(zero, name="zero"))
    sq = helper.make_node("ReduceSum", ["cls_emb"], ["sq"], axes=[1], keepdims=1)
    eps = const("eps", np.array(1e-12, dtype=np.float32))
    add_eps = helper.make_node("Add", ["sq", eps.output[0]], ["add_eps"])
    norm = helper.make_node("Sqrt", ["add_eps"], ["norm"])
    img_feat = helper.make_node("Div", ["cls_emb", "norm"], ["img_feat"])
    nodes.extend([cls_emb, sq, eps, add_eps, norm, img_feat])
    return nodes, init, "img_feat"

# ---------- Text encoder ----------
def build_text_encoder(txt):
    nodes, init = [], []
    # embedding
    W_txt = rand(VOCAB, D_TXT)
    b_txt = rand(D_TXT)
    init.extend([nh.from_array(W_txt, name="W_txt"), nh.from_array(b_txt, name="b_txt")])
    emb = helper.make_node("Gather", ["W_txt", txt], ["emb"])
    emb_b = helper.make_node("Add", ["emb", "b_txt"], ["emb_b"])
    # positional
    P_txt = rand(SEQ_TXT, D_TXT)
    init.append(nh.from_array(P_txt, name="P_txt"))
    seq = helper.make_node("Add", ["emb_b", "P_txt"], ["seq"])
    nodes.extend([emb, emb_b, seq])
    # project to 768
    Wproj = rand(D_TXT, D_IMG)
    init.append(nh.from_array(Wproj, name="Wproj_txt"))
    seq_proj = helper.make_node("MatMul", ["seq", "Wproj_txt"], ["seq_proj"])
    nodes.append(seq_proj)
    # 12 layers
    h = "seq_proj"
    for lyr in range(N_LAYERS):
        blk_nodes, h = transformer_block(h, f"txt_l{lyr}")
        nodes.extend(blk_nodes)
    # mean pool
    txt_emb = helper.make_node("ReduceMean", [h], ["txt_emb"], axes=[1])
    # L2 norm
    sq_txt = helper.make_node("ReduceSum", ["txt_emb"], ["sq_txt"], axes=[1], keepdims=1)
    eps_txt = const("eps_txt", np.array(1e-12, dtype=np.float32))
    add_eps_txt = helper.make_node("Add", ["sq_txt", eps_txt.output[0]], ["add_eps_txt"])
    norm_txt = helper.make_node("Sqrt", ["add_eps_txt"], ["norm_txt"])
    txt_feat = helper.make_node("Div", ["txt_emb", "norm_txt"], ["txt_feat"])
    nodes.extend([txt_emb, sq_txt, eps_txt, add_eps_txt, norm_txt, txt_feat])
    return nodes, init, "txt_feat"

# ---------- Full graph ----------
def build_model():
    # inputs
    img = helper.make_tensor_value_info("img", TensorProto.FLOAT, [1,3,IMG_SIZE,IMG_SIZE])
    txt = helper.make_tensor_value_info("txt", TensorProto.INT64, [1,SEQ_TXT])
    # build encoders
    img_nodes, img_init, img_out = build_image_encoder("img")
    txt_nodes, txt_init, txt_out = build_text_encoder("txt")
    # similarity
    sim = helper.make_node("MatMul", [img_out, txt_out], ["sim"])
    # output
    sim_out = helper.make_tensor_value_info("sim", TensorProto.FLOAT, [1,1])
    # assemble
    g = helper.make_graph(
        img_nodes + txt_nodes + [sim],
        "clip_py",
        [img, txt],
        [sim_out],
        img_init + txt_init
    )
    m = helper.make_model(g, producer_name="fuse-example", opset_imports=[helper.make_opsetid("", OPSSET)])
    # metadata
    helper.set_model_props(m, {
        "fuse": FUSE,
        "domain": DOMAIN,
        "id": ID_URL
    })
    return m

# ---------- Export ----------
if __name__ == "__main__":
    model = build_model()
    onnx.save(model, "golden/clip_py.onnx")
    print("✅ golden clip_py.onnx:", {p.key:p.value for p in model.metadata_props})
