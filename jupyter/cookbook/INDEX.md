# Fuse Jupyter Cookbook Index 🍳

A curated collection of practical examples, recipes, and patterns for building ONNX models with Fuse in Jupyter notebooks.

---

## 📚 Browse by Category

### 🎯 Getting Started (5 recipes)
| Recipe | Description | Difficulty | Time |
|--------|-------------|------------|------|
| [Hello Fuse](01_hello_fuse.ipynb) | Your first Fuse model | ⭐ Beginner | 5 min |
| [Basic Operations](02_basic_ops.ipynb) | Common tensor operations | ⭐ Beginner | 10 min |
| [Type System](03_types.ipynb) | Working with types and shapes | ⭐⭐ Beginner | 15 min |
| [Running Models](04_run_models.ipynb) | Inference and validation | ⭐ Beginner | 10 min |
| [Debugging](05_debugging.ipynb) | Common issues and solutions | ⭐⭐ Intermediate | 20 min |

---

### 🏗️ Model Building (8 recipes)
| Recipe | Description | Difficulty | Time |
|--------|-------------|------------|------|
| [Linear Layers](10_linear.ipynb) | Fully connected layers | ⭐ Beginner | 15 min |
| [Convolutions](11_convolutions.ipynb) | Conv2d, Conv3d, padding | ⭐⭐ Intermediate | 25 min |
| [Pooling](12_pooling.ipynb) | MaxPool, AvgPool, GlobalPool | ⭐ Beginner | 15 min |
| [Activations](13_activations.ipynb) | ReLU, Sigmoid, GELU, etc. | ⭐ Beginner | 10 min |
| [Normalization](14_normalization.ipynb) | BatchNorm, LayerNorm, InstanceNorm | ⭐⭐ Intermediate | 30 min |
| [Dropout](15_dropout.ipynb) | Training-time regularization | ⭐⭐ Intermediate | 20 min |
| [Residual Connections](16_residual.ipynb) | Skip connections and ResNet blocks | ⭐⭐ Intermediate | 25 min |
| [Multi-Head Attention](17_attention.ipynb) | Transformer attention mechanism | ⭐⭐⭐ Advanced | 45 min |

---

### 🧠 Neural Network Architectures (6 recipes)
| Recipe | Description | Difficulty | Time |
|--------|-------------|------------|------|
| [Simple MLP](20_mlp.ipynb) | Multi-layer perceptron | ⭐ Beginner | 20 min |
| [LeNet-5](21_lenet.ipynb) | Classic CNN architecture | ⭐⭐ Intermediate | 30 min |
| [ResNet-18](22_resnet.ipynb) | Deep residual network | ⭐⭐⭐ Advanced | 45 min |
| [VGG](23_vgg.ipynb) | VGG-style networks | ⭐⭐ Intermediate | 35 min |
| [Transformer](24_transformer.ipynb) | Full transformer encoder | ⭐⭐⭐ Advanced | 60 min |
| [U-Net](25_unet.ipynb) | Segmentation architecture | ⭐⭐⭐ Advanced | 50 min |

---

### 📊 Data & Preprocessing (5 recipes)
| Recipe | Description | Difficulty | Time |
|--------|-------------|------------|------|
| [Load NumPy Data](30_numpy_data.ipynb) | Working with .npz files | ⭐ Beginner | 10 min |
| [Image Preprocessing](31_image_prep.ipynb) | Resize, normalize, augment | ⭐⭐ Intermediate | 25 min |
| [Custom Embeddings](32_embeddings.ipynb) | Load pretrained embeddings | ⭐⭐ Intermediate | 30 min |
| [HuggingFace Integration](33_huggingface.ipynb) | Import HF model weights | ⭐⭐⭐ Advanced | 40 min |
| [External Data](34_external_data.ipynb) | Large tensors with `--externalize` | ⭐⭐ Intermediate | 20 min |

---

### 🎓 Training & Optimization (7 recipes)
| Recipe | Description | Difficulty | Time |
|--------|-------------|------------|------|
| [Trainable Parameters](40_trainable.ipynb) | Mark params for training | ⭐ Beginner | 15 min |
| [Loss Functions](41_losses.ipynb) | CrossEntropy, MSE, custom losses | ⭐⭐ Intermediate | 25 min |
| [Optimizers](42_optimizers.ipynb) | SGD, Adam, Adagrad | ⭐⭐ Intermediate | 30 min |
| [Learning Rate Schedules](43_lr_schedules.ipynb) | Warmup, decay, cyclic | ⭐⭐⭐ Advanced | 35 min |
| [Gradient Clipping](44_grad_clip.ipynb) | Prevent exploding gradients | ⭐⭐ Intermediate | 20 min |
| [Training Loops](45_training_loop.ipynb) | Full training example | ⭐⭐⭐ Advanced | 50 min |
| [Fine-Tuning](46_finetuning.ipynb) | Transfer learning patterns | ⭐⭐⭐ Advanced | 45 min |

---

### 🔧 Advanced Topics (8 recipes)
| Recipe | Description | Difficulty | Time |
|--------|-------------|------------|------|
| [Dynamic Shapes](50_dynamic.ipynb) | Variable batch sizes | ⭐⭐ Intermediate | 30 min |
| [Control Flow](51_control_flow.ipynb) | If/Loop operators | ⭐⭐⭐ Advanced | 40 min |
| [Custom Operators](52_custom_ops.ipynb) | Extend ONNX with custom ops | ⭐⭐⭐ Advanced | 60 min |
| [Quantization](53_quantization.ipynb) | INT8 quantization | ⭐⭐⭐ Advanced | 45 min |
| [Model Fusion](54_fusion.ipynb) | Import and combine models | ⭐⭐ Intermediate | 35 min |
| [Graph Optimization](55_optimization.ipynb) | Optimize ONNX graphs | ⭐⭐⭐ Advanced | 40 min |
| [Mixed Precision](56_mixed_precision.ipynb) | FP16/BF16 training | ⭐⭐⭐ Advanced | 45 min |
| [Distributed Training](57_distributed.ipynb) | Multi-GPU strategies | ⭐⭐⭐ Advanced | 60 min |

---

### 🚀 Deployment (6 recipes)
| Recipe | Description | Difficulty | Time |
|--------|-------------|------------|------|
| [Export Models](60_export.ipynb) | Save to `.onnx` format | ⭐ Beginner | 10 min |
| [ONNX Runtime](61_onnx_runtime.ipynb) | Run with ONNXRuntime | ⭐⭐ Intermediate | 25 min |
| [WebAssembly](62_wasm.ipynb) | Deploy to browser | ⭐⭐⭐ Advanced | 50 min |
| [Mobile (iOS/Android)](63_mobile.ipynb) | Mobile deployment | ⭐⭐⭐ Advanced | 45 min |
| [TensorRT](64_tensorrt.ipynb) | GPU optimization | ⭐⭐⭐ Advanced | 40 min |
| [Model Serving](65_serving.ipynb) | REST API with FastAPI | ⭐⭐⭐ Advanced | 50 min |

---

### 🎨 Computer Vision (7 recipes)
| Recipe | Description | Difficulty | Time |
|--------|-------------|------------|------|
| [Image Classification](70_classification.ipynb) | Full classification pipeline | ⭐⭐ Intermediate | 40 min |
| [Image Visualization](visuals_image.ipynb) | Inspect tensors as images with `%image` and `%inspect` | ⭐ Beginner | 10 min |
| [Object Detection](71_detection.ipynb) | Bounding box prediction | ⭐⭐⭐ Advanced | 60 min |
| [Semantic Segmentation](72_segmentation.ipynb) | Pixel-wise classification | ⭐⭐⭐ Advanced | 55 min |
| [Style Transfer](73_style_transfer.ipynb) | Neural style transfer | ⭐⭐⭐ Advanced | 50 min |
| [Attention Visualization](visuals_attention.ipynb) | Visualize attention heatmaps with `%attention` | ⭐ Beginner | 10 min |
| [GANs](74_gans.ipynb) | Generative adversarial networks | ⭐⭐⭐ Advanced | 70 min |

---

### 📝 Natural Language Processing (5 recipes)
| Recipe | Description | Difficulty | Time |
|--------|-------------|------------|------|
| [Word Embeddings](80_embeddings.ipynb) | Word2Vec, GloVe | ⭐⭐ Intermediate | 30 min |
| [Sequence Models](81_sequence.ipynb) | LSTM, GRU patterns | ⭐⭐⭐ Advanced | 45 min |
| [Text Classification](82_text_class.ipynb) | Sentiment analysis | ⭐⭐ Intermediate | 40 min |
| [Named Entity Recognition](83_ner.ipynb) | Token classification | ⭐⭐⭐ Advanced | 50 min |
| [BERT Fine-Tuning](84_bert.ipynb) | Pretrained transformer | ⭐⭐⭐ Advanced | 60 min |

---

### 🧪 Experimental Features (4 recipes)
| Recipe | Description | Difficulty | Time |
|--------|-------------|------------|------|
| [TensorBus](90_tensorbus.ipynb) | Distributed tensor sync (CRDT) | ⭐⭐⭐ Advanced | 45 min |
| [LLM Assistance](91_llm_help.ipynb) | Using Copilot Chat | ⭐ Beginner | 15 min |
| [Auto-completion](92_autocomplete.ipynb) | Context-aware IDE features | ⭐ Beginner | 10 min |
| [Interactive Debugging](93_debug.ipynb) | Rich error cards | ⭐⭐ Intermediate | 20 min |

---

## 🔍 Browse by Use Case

### "I want to..."

- **Get started with Fuse** → [Hello Fuse](01_hello_fuse.ipynb)
- **Build an image classifier** → [Image Classification](70_classification.ipynb)
- **Train a model from scratch** → [Training Loops](45_training_loop.ipynb)
- **Import PyTorch weights** → [HuggingFace Integration](33_huggingface.ipynb)
- **Deploy to production** → [Model Serving](65_serving.ipynb)
- **Optimize for mobile** → [Mobile Deployment](63_mobile.ipynb)
- **Build a transformer** → [Transformer](24_transformer.ipynb)
- **Debug model issues** → [Debugging](05_debugging.ipynb)

---

## 📈 Learning Paths

### Path 1: Beginner to Practitioner (10 recipes, ~4 hours)
1. [Hello Fuse](01_hello_fuse.ipynb)
2. [Basic Operations](02_basic_ops.ipynb)
3. [Type System](03_types.ipynb)
4. [Linear Layers](10_linear.ipynb)
5. [Activations](13_activations.ipynb)
6. [Simple MLP](20_mlp.ipynb)
7. [Trainable Parameters](40_trainable.ipynb)
8. [Loss Functions](41_losses.ipynb)
9. [Export Models](60_export.ipynb)
10. [ONNX Runtime](61_onnx_runtime.ipynb)

### Path 2: Computer Vision Expert (8 recipes, ~6 hours)
1. [Convolutions](11_convolutions.ipynb)
2. [Pooling](12_pooling.ipynb)
3. [Normalization](14_normalization.ipynb)
4. [ResNet-18](22_resnet.ipynb)
5. [Image Preprocessing](31_image_prep.ipynb)
6. [Image Classification](70_classification.ipynb)
7. [Object Detection](71_detection.ipynb)
8. [Model Serving](65_serving.ipynb)

### Path 3: NLP Specialist (7 recipes, ~5 hours)
1. [Word Embeddings](80_embeddings.ipynb)
2. [Multi-Head Attention](17_attention.ipynb)
3. [Transformer](24_transformer.ipynb)
4. [Text Classification](82_text_class.ipynb)
5. [BERT Fine-Tuning](84_bert.ipynb)
6. [Custom Embeddings](32_embeddings.ipynb)
7. [Quantization](53_quantization.ipynb)

---

## 🏷️ Tags

Browse recipes by tags:

- `#beginner` (15 recipes)
- `#intermediate` (22 recipes)
- `#advanced` (26 recipes)
- `#computer-vision` (12 recipes)
- `#nlp` (8 recipes)
- `#training` (10 recipes)
- `#deployment` (7 recipes)
- `#optimization` (5 recipes)
- `#experimental` (4 recipes)

---

## 💡 Contributing New Recipes

Have a recipe to share? Follow these guidelines:

### Recipe Template

```python
"""
Title: My Awesome Recipe
Category: Model Building
Difficulty: Intermediate
Time: 30 minutes
Tags: #intermediate #computer-vision

Description:
Learn how to build [specific thing] using Fuse.

Prerequisites:
- Basic Fuse knowledge
- Understanding of [specific concept]

What You'll Learn:
1. [Key concept 1]
2. [Key concept 2]
3. [Key concept 3]
"""

# Import dependencies
%load_ext src.jupyter.magics
%load_ext src.jupyter.inspect.magics
import numpy as np

# Step-by-step implementation with explanations...
```

### Submission Process

1. Fork repository
2. Create notebook in `jupyter/cookbook/`
3. Add entry to this index
4. Test thoroughly
5. Submit PR with tag `[cookbook]`

---

## 📊 Statistics

- **Total Recipes:** 63
- **Beginner:** 15 (24%)
- **Intermediate:** 22 (35%)
- **Advanced:** 26 (41%)
- **Total Learning Time:** ~45 hours
- **Most Popular:** [Image Classification](70_classification.ipynb)
- **Most Advanced:** [Distributed Training](57_distributed.ipynb)

---

## 🎯 Quick Search

Use Cmd/Ctrl+F to search for:
- Specific operators (e.g., "Conv", "Attention")
- Techniques (e.g., "normalization", "quantization")
- Architectures (e.g., "ResNet", "Transformer")
- Use cases (e.g., "classification", "detection")

---

**Last Updated:** February 4, 2026  
**Maintainers:** Fuse Team  
**Contributions:** 12 contributors
